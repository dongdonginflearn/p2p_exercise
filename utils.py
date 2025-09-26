from PIL import Image, ImageDraw, ImageFont


def concatenate_images_with_spacing(images, spacing=20):
    """
    Concatenates a list of PIL images horizontally with a specified spacing between them.

    Args:
        images (list): A list of PIL.Image objects.
        spacing (int): The spacing (in pixels) between each image.

    Returns:
        PIL.Image: A new image with the input images concatenated horizontally.
    """
    # Calculate the total width and maximum height of the final image
    total_width = sum(img.width for img in images) + spacing * (len(images) - 1)
    max_height = max(img.height for img in images)
    
    # Create a new blank image with the calculated dimensions
    result_image = Image.new("RGB", (total_width, max_height), (255, 255, 255))  # White background
    
    # Paste each image into the result image with spacing
    x_offset = 0
    for img in images:
        result_image.paste(img, (x_offset, 0))
        x_offset += img.width + spacing
    
    return result_image


def append_text_below(
    image: Image.Image,
    text: str,
    *,
    font_path: str | None = None,
    font_size: int | None = None,
    text_color=(0, 0, 0),
    bg_color=(255, 255, 255),
    padding: int = 16,
    align: str = "center",
    line_spacing: float = 1.3,
) -> Image.Image:
    """
    원본 이미지 하단에 텍스트를 붙여 새 이미지를 반환합니다.

    Parameters
    ----------
    image : PIL.Image
        원본 이미지
    text : str
        하단에 넣을 문자열(개행 문자 포함 가능)
    font_path : str | None
        사용할 .ttf 폰트 경로 (None이면 DejaVuSans 시도 후 기본 폰트)
    font_size : int | None
        폰트 크기 (None이면 이미지 폭 기준 자동 결정)
    text_color : tuple
        텍스트 색 (RGB 또는 RGBA)
    bg_color : tuple | None
        하단 배경색 (RGB 또는 RGBA). RGBA 이미지를 투명 배경으로 만들고 싶다면 None을 주고
        image가 RGBA일 때 자동으로 완전 투명(0) 배경을 사용합니다.
    padding : int
        텍스트 영역 내부 여백(px)
    align : str
        "left" | "center" | "right"
    line_spacing : float
        줄간 간격 배수 (1.0 이상 권장)

    Returns
    -------
    PIL.Image
        텍스트가 아래에 붙은 새 이미지
    """
    if not isinstance(image, Image.Image):
        raise TypeError("image 는 PIL.Image.Image 객체여야 합니다.")

    # 1) 출력 모드 결정 (투명도 보존)
    has_alpha = (image.mode in ("RGBA", "LA")) or ("transparency" in image.info)
    out_mode = "RGBA" if has_alpha else "RGB"
    base = image.convert(out_mode)
    W = base.width

    # 2) 폰트 로딩
    if font_size is None:
        font_size = max(12, min(int(W * 0.5), 72))  # 이미지 폭에 비례해 적당히
    font = None
    try:
        if font_path:
            font = ImageFont.truetype(font_path, font_size)
        else:
            # 시스템 기본 DejaVuSans 시도
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except Exception:
        font = ImageFont.load_default(font_size)

    # 3) 줄바꿈(워드랩)
    #    이미지 폭 - 좌우 패딩을 초과하지 않도록 단어 단위로 개행
    draw_tmp = ImageDraw.Draw(base)
    max_text_width = max(1, W - padding * 2)

    lines = []
    for paragraph in text.split("\n"):
        words = paragraph.split(" ")
        line = ""
        for w in words:
            cand = w if line == "" else line + " " + w
            # bbox: (l, t, r, b)
            r = draw_tmp.textbbox((0, 0), cand, font=font)
            if (r[2] - r[0]) <= max_text_width:
                line = cand
            else:
                if line:     # 기존 라인을 확정하고 새 라인 시작
                    lines.append(line)
                    line = w
                else:
                    # 단일 단어가 너무 길어도 그냥 넣음(폭 초과 허용)
                    lines.append(w)
                    line = ""
        if line:
            lines.append(line)

    # 4) 텍스트 전체 bbox 계산
    spacing_px = max(0, int(font.size * (line_spacing - 1.0)))
    all_text = "\n".join(lines) if lines else ""
    # 빈 문자열일 때 bbox가 None이 되지 않도록 한 글자 높이를 기준으로 처리
    if all_text.strip() == "":
        text_height = 0
        text_width = 0
    else:
        bbox = draw_tmp.multiline_textbbox((0, 0), all_text, font=font, spacing=spacing_px, align=align)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

    # 5) 출력 캔버스 생성 (하단 배경색 결정)
    if bg_color is None and out_mode == "RGBA":
        footer_bg = (0, 0, 0, 0)  # 완전 투명
    else:
        # 모드에 맞게 길이 보정
        if out_mode == "RGBA":
            if len(bg_color) == 3:
                footer_bg = (*bg_color, 255)
            else:
                footer_bg = bg_color
        else:
            # RGB 모드
            if len(bg_color) == 4:
                footer_bg = bg_color[:3]
            else:
                footer_bg = bg_color

    footer_h = (padding * 2 + text_height) if text_height > 0 else 0
    out = Image.new(out_mode, (W, base.height + footer_h), footer_bg if footer_h > 0 else None)

    # 6) 원본 붙이기
    out.paste(base, (0, 0))

    # 7) 텍스트 그리기
    if text_height > 0:
        draw = ImageDraw.Draw(out)
        # 정렬에 따른 X 좌표
        if align == "left":
            x = padding
        elif align == "right":
            x = max(padding, W - padding - text_width)
        else:  # center
            x = max(padding, (W - text_width) // 2)

        y = base.height + padding

        # 텍스트 색도 모드에 맞게 보정
        if out_mode == "RGBA":
            if len(text_color) == 3:
                fill = (*text_color, 255)
            else:
                fill = text_color
        else:
            fill = text_color[:3] if len(text_color) == 4 else text_color

        draw.multiline_text((x, y), all_text, font=font, fill=fill, spacing=spacing_px, align=align)

    return out


def align_token_indices(tokens_of_source_prompt, tokens_of_target_prompt, max_len=77):
    target_token_indices = []

    max_index_of_source_prompt = 0
    for target_token in tokens_of_target_prompt:
        try:
            token_index_of_source_prompt = tokens_of_source_prompt.index(target_token)
            if token_index_of_source_prompt > max_index_of_source_prompt:
                max_index_of_source_prompt = token_index_of_source_prompt
        except:
            token_index_of_source_prompt = None
        target_token_indices.append(token_index_of_source_prompt)

    # padding
    while len(target_token_indices) < max_len:
        max_index_of_source_prompt += 1
        target_token_indices.append(max_index_of_source_prompt)

    return target_token_indices
