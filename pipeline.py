import torch
from diffusers import StableDiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor, Attention
from typing import Optional


class Promp2PromptPipeline(StableDiffusionPipeline):
    def __call__(self, *args, **kwargs):
        controller = kwargs.pop("controller", None)
        if controller == None:
            raise KeyError("No controller!!")

        self.register_attention_control(controller)

        return super().__call__(*args, **kwargs)
    
    def register_attention_control(self, controller):
        attn_procs = {}

        cnt_cross_attn_layers = 0
        for name in self.unet.attn_processors.keys():
            if name.endswith("attn2.processor"):
                cnt_cross_attn_layers += 1
                attn_procs[name] = P2PCrossAttnProcessor(controller, name)
            else:
                attn_procs[name] = AttnProcessor()

        self.unet.set_attn_processor(attn_procs)
        controller.set_num_cross_attn_layers(cnt_cross_attn_layers)

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        latents = super().prepare_latents(batch_size, num_channels_latents, height, width, dtype, device, generator, latents)

        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if batch_size > 1:
            latents = latents[0].expand(shape)

        return latents


class P2PCrossAttnProcessor(AttnProcessor):
    def __init__(self, controller, attn_name):
        super().__init__()
        self.controller = controller
        self.attn_name = attn_name

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # one line change
        attention_probs = self.controller(attention_probs, self.attn_name)
        
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
