"""이미지 파일 속성 메타(크기·색모드·포맷·대표색) 추출 — ``processing/skills/image_skill.py`` 가 호출.

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
    """RGB 값을 ``#rrggbb`` 문자열로 바꾼다(메타에 담을 형태).

    Args:
        rgb: 0~255 세 값.

    Returns:
        16진수 색 문자열.
    """
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _extract_dominant_colors(img: Image.Image, top_k: int = 5) -> list[str]:
    """이미지의 대표 색을 뽑는다.

    원본 그대로 세면 픽셀 수만큼 오래 걸린다 — 작게 줄여도 색 분포는 거의 그대로다.

    Args:
        img: 대상 이미지.
        top_k: 뽑을 색 수. **0 이하면 빈 목록**을 돌려준다(색 정보를 끄는 용도).

    Returns:
        16진수 색 문자열 목록(많이 쓰인 색 순서).
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
    """이미지의 크기·색 모드·포맷과 대표 색을 뽑는다.

    Args:
        file_path: 대상 파일.
        dominant_top_k: 뽑을 대표 색 수. 0 이하면 색을 뽑지 않는다.

    Returns:
        폭·높이·색 모드·포맷·대표 색 목록.

    Raises:
        FileNotFoundError: 파일이 없을 때.
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

