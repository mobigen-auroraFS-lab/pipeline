"""이미지 파일 속성 메타(크기·색모드·포맷·대표색) 추출 — ``src/skills/image_skill.py`` 가 호출.

Pillow 만 쓰는 경량 추출이다. 의미 정보(캡션·키워드·라벨)와 CLIP 벡터는 skill 쪽에서
VLM·CLIP 으로 별도 생성한다 — 이 모듈은 그와 무관한 결정적 파일 속성만 담당한다.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from PIL import Image


class ImageMeta(TypedDict):
    width: int
    height: int
    color_mode: str
    file_format: str
    dominant_colors: list[str]


def _to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _extract_dominant_colors(img: Image.Image, top_k: int = 5) -> list[str]:
    """
    Pillow quantize 기반으로 대표 색상 추출.
    속도를 위해 512x512 이하로 리사이즈 후 계산한다.
    """
    if top_k <= 0:
        return []

    # 팔레트 추출은 RGB가 가장 다루기 쉽다.
    rgb_img = img.convert("RGB")
    rgb_img.thumbnail((512, 512))
    quantized = rgb_img.quantize(colors=top_k, method=Image.Quantize.MEDIANCUT)

    palette = quantized.getpalette() or []
    counts = quantized.getcolors(maxcolors=top_k * 8) or []

    result: list[str] = []
    for _, palette_idx in sorted(counts, key=lambda x: x[0], reverse=True)[:top_k]:
        base = palette_idx * 3
        if base + 2 >= len(palette):
            continue
        rgb = (
            int(palette[base]),
            int(palette[base + 1]),
            int(palette[base + 2]),
        )
        result.append(_to_hex(rgb))
    return result


def extract_image_meta(
    file_path: str | Path,
    *,
    dominant_top_k: int = 5
) -> ImageMeta:
    """
    이미지 메타 및 특징 정보 추출.

    Returns
    -------
    ImageMeta
        ``width``, ``height``, ``color_mode``, ``file_format``, ``dominant_colors``(hex 문자열 리스트).
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(str(path))

    with Image.open(path) as img:
        width, height = img.size
        color_mode = img.mode
        file_format = (img.format or path.suffix.replace(".", "")).lower()
        dominant_colors = _extract_dominant_colors(img, top_k=dominant_top_k)

    return {
        "width": int(width),
        "height": int(height),
        "color_mode": str(color_mode),
        "file_format": str(file_format),
        "dominant_colors": dominant_colors
    }

