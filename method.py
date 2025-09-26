import torch
import numpy as np
from PIL import Image
from utils import concatenate_images_with_spacing, append_text_below, align_token_indices
import abc


class AttentionStore:
    """각 step동안 UNet 레이어들의 cross-attention map을 저장"""
    
    def __init__(self, prompts, tokenizer, res=16, token_length=77):
        self.prompts = prompts
        if len(self.prompts) > 1:
            raise ValueError("num of prompts should be 1")
        self.encoder = tokenizer.encode
        self.decoder = tokenizer.decode
        self.tokens = self.encoder(self.prompts[0])
        self.res = res
        self.token_length = token_length
        self.cross_attn_maps = {}
        self.num_cross_attn_layers = -1 # to be set

    def set_num_cross_attn_layers(self, num_cross_attn_layers):
        self.num_cross_attn_layers = num_cross_attn_layers

    def __call__(self, attn, attn_name):
        if attn.shape[1] != self.res ** 2:
            return attn

        uncond_attn, cond_attn = attn.chunk(2) 
        cond_attn = cond_attn.reshape(-1, self.res, self.res, self.token_length)

        if not attn_name in self.cross_attn_maps.keys():
            self.cross_attn_maps[attn_name] = cond_attn
        else:
            self.cross_attn_maps[attn_name] += cond_attn
        return attn # no edit

    def get_average_cross_attention(self):
        out = []
        for item in self.cross_attn_maps.values():
            out.append(item)
        out = torch.cat(out, dim=0)
        out = out.mean(dim=0).cpu()

        images = []
        for i in range(len(self.tokens)):
            image = out[:,:,i]
            image = 255 * image / image.max()
            image = image.unsqueeze(-1).expand(*image.shape, 3)
            image = image.numpy().astype(np.uint8)
            image = Image.fromarray(image).resize((256, 256))
            text = self.decoder((int(self.tokens[i])))
            image_with_text = append_text_below(image, text, font_size=36)
            images.append(image_with_text)
        return concatenate_images_with_spacing(images)


class AttentionEdit(abc.ABC):
    def __init__(self, prompts, tokenizer, total_steps, replace_ratio):
        self.prompts = prompts
        if len(self.prompts) != 2:
            raise ValueError("num of prompts should be 2")
        self.encoder = tokenizer.encode
        self.decoder = tokenizer.decode
        self.cur_step = total_steps
        self.tau = (1 - replace_ratio) * total_steps
        self.cnt_cross_attn_layers = 0
        self.num_cross_attn_layers = -1 # to be set
    
    @abc.abstractmethod
    def edit(self, attn_base, attn_target):
        raise NotImplementedError

    def set_num_cross_attn_layers(self, num_cross_attn_layers):
        self.num_cross_attn_layers = num_cross_attn_layers

    def __call__(self, attn, attn_name):
        if self.cnt_cross_attn_layers == self.num_cross_attn_layers:
            self.cur_step -= 1
            self.cnt_cross_attn_layers = 0

        uncond_attn, cond_attn = attn.chunk(2)
        attn_base, attn_target = cond_attn.chunk(2)

        attn_new = self.edit(attn_base, attn_target)
        attn = torch.cat((uncond_attn, attn_base, attn_new))

        self.cnt_cross_attn_layers += 1
        return attn


class AttentionReplace(AttentionEdit):
    def edit(self, attn_base, attn_target):
        if self.cur_step < self.tau:
            return attn_target
        else:
            return attn_base


class AttentionRefine(AttentionEdit):
    def edit(self, attn_base, attn_target):
        if self.cur_step < self.tau:
            return attn_target
        else:
            prompt_base = self.prompts[0]
            prompt_target = self.prompts[1]

            token_base = self.encoder(prompt_base)
            token_target = self.encoder(prompt_target)

            A = align_token_indices(token_base, token_target)

            attn_new = torch.empty_like(attn_target)
            for j in range(attn_target.shape[2]):
                if A[j] == None:
                    attn_new[:,:,j] = attn_target[:,:,j]
                else:
                    attn_new[:,:,j] = attn_base[:,:,A[j]]
            return attn_new


class AttentionReweight(AttentionEdit):
    def __init__(self, prompts, tokenizer, total_steps, replace_ratio,
                 target_word, weight):
        super().__init__(prompts, tokenizer, total_steps, replace_ratio)
        self.target_word = target_word
        self.c = weight

    def edit(self, attn_base, attn_target):
        if self.cur_step < self.tau:
            return attn_target
        else:
            prompt_target = self.prompts[1]
            tokens_target = self.encoder(prompt_target)
            token_of_target_word = self.encoder(self.target_word)[1]

            j_prime = tokens_target.index(token_of_target_word)

            attn_new = torch.empty_like(attn_target)
            for j in range(attn_target.shape[2]):
                if j == j_prime:
                    attn_new[:,:,j] = self.c * attn_base[:,:,j]
                else:
                    attn_new[:,:,j] = attn_base[:,:,j]
            return attn_new
