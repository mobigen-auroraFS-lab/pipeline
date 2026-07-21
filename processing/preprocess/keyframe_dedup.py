"""048 — 영상 키프레임 near-duplicate 제거 순수 코어(VLM 전 결정적 2단계 dedup).

JPEG 추출 직후·VLM 이전에 시각적으로 거의 동일한 키프레임을 결정적 2단계로 걸러낸다:

  1) **1차 dHash(perceptual hash)** — 64-bit. 비교 keep 집합 중 Hamming ≤ ``hash_max`` 면 near-dup
     후보(빠른 CPU 필터).
  2) **2차 SSIM/HSV** — 후보만 동일 resize 규격으로 SSIM. ``≥ ssim_min`` 이면 중복 확정(skip).
     SSIM ∈ [``ssim_gray_lo``, ``ssim_min``) 애매 구간에서만 HSV histogram correlation 보조 판정
     (``≥ hist_min`` 이면 skip). 히스토그램 단독 skip 은 금지(FR-304) — 반드시 1차 후보 통과 후에만.

본 모듈은 **순수**하다: IO(파일)·설정(settings)·LLM 호출이 전혀 없고, ``numpy``/``cv2`` 만으로
계산한다. 학습·파인튜닝·난수가 없고(헌법 1조), 정수 XOR/popcount·고정 resize·표준 SSIM 식으로
**동일 입력·설정 → 동일 keep 목록**을 보장한다(헌법 3조 결정성). 신규 의존성 0(``imagehash``·
``scikit-image`` 미사용 — SSIM·HSV correlation 도 cv2/numpy 직접 구현).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

# 069 D2: dedup 기본값 단일 출처(settings env fallback 과 공유하는 경량 상수 모듈·cv2 무의존).
from src.config.keyframe_dedup_defaults import (
    DEFAULT_COMPARE_MODE,
    DEFAULT_HASH_MAX,
    DEFAULT_HIST_MIN,
    DEFAULT_RECENT_WINDOW,
    DEFAULT_SSIM_GRAY_LO,
    DEFAULT_SSIM_MIN,
)

if TYPE_CHECKING:
    from src.embedders.frame_types import KeyframeBytesResult

# dHash: (HASH_SIZE+1) × HASH_SIZE grayscale 의 인접 열 비교 → HASH_SIZE² = 64 bit.
HASH_SIZE = 8
# SSIM·HSV 비교용 정사각 규격(소형·CPU 절감, 결정성 고정값).
SSIM_SIZE = 128

# 지원 비교 모드(화이트리스트 — 오설정 fail-fast).
_COMPARE_MODES = frozenset({"recent", "last", "global"})


@dataclass(frozen=True)
class KeyframeDedupConfig:
    """dedup 임계·모드 설정(frozen — 동일 설정 → 동일 결과 결정성 보장, 헌법 3조).

    settings(``VIDEO_KEYFRAME_DEDUP_*``)에서 video_skill 이 빌드해 주입한다(단일 출처·FR-501).
    오설정(모드·범위)은 ``__post_init__`` 에서 즉시 ``ValueError`` 로 차단한다 — 잘못된 값이 조용히
    dedup 을 무력화(예: hash_max<0)하거나 과대 적용하지 않도록(레포 fail-fast 관례).
    """

    enabled: bool  # 필수(video_skill 이 settings 에서 항상 주입) — 기본값 없음
    hash_max: int = DEFAULT_HASH_MAX
    ssim_min: float = DEFAULT_SSIM_MIN
    ssim_gray_lo: float = DEFAULT_SSIM_GRAY_LO
    hist_min: float = DEFAULT_HIST_MIN
    compare_mode: str = DEFAULT_COMPARE_MODE  # "recent" | "last" | "global"
    recent_window: int = DEFAULT_RECENT_WINDOW

    def __post_init__(self) -> None:
        if self.compare_mode not in _COMPARE_MODES:
            raise ValueError(
                f"compare_mode 는 {sorted(_COMPARE_MODES)} 중 하나여야 함: {self.compare_mode!r}"
            )
        if self.hash_max < 0:
            raise ValueError(f"hash_max 는 0 이상이어야 함: {self.hash_max}")
        if self.recent_window < 1:
            raise ValueError(f"recent_window 는 1 이상이어야 함: {self.recent_window}")
        for _name in ("ssim_min", "ssim_gray_lo", "hist_min"):
            _v = getattr(self, _name)
            if not 0.0 <= _v <= 1.0:
                raise ValueError(f"{_name} 는 0..1 범위여야 함: {_v}")
        if self.ssim_gray_lo > self.ssim_min:
            raise ValueError(
                f"ssim_gray_lo({self.ssim_gray_lo}) 는 ssim_min({self.ssim_min}) 이하여야 함"
            )


def hamming(a: int, b: int) -> int:
    """두 64-bit 해시의 Hamming distance — 정수 XOR + popcount(결정적, FR-203)."""
    return bin(a ^ b).count("1")


def dhash(jpeg_bytes: bytes) -> int:
    """JPEG → 64-bit dHash perceptual hash(FR-201).

    grayscale 디코드 → (9×8) 소형 resize 후 **인접 열 밝기 비교**(좌<우)를 비트로 누적한다.
    resize·비교가 결정적이라 동일 bytes → 동일 hash, 시각적 미세 변형 → 작은 Hamming 거리다.
    """
    img = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("dhash: JPEG 디코드 실패")
    small = cv2.resize(img, (HASH_SIZE + 1, HASH_SIZE), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]  # (8,8) — 인접 열 밝기 비교
    bits = 0
    for v in diff.flatten():
        bits = (bits << 1) | int(v)
    return bits


def _decode_gray(jpeg_bytes: bytes, size: int = SSIM_SIZE) -> np.ndarray:
    """JPEG → grayscale ``size×size`` float64(SSIM 입력). 고정 resize 로 결정적."""
    img = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("_decode_gray: JPEG 디코드 실패")
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA).astype(np.float64)


def ssim(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    """Wang et al. 윈도 SSIM 평균(0..1)(FR-301·P7).

    cv2.GaussianBlur(11×11, σ=1.5)로 국소 평균·분산·공분산을 구하고 표준 C1/C2(=(0.01·255)²,
    (0.03·255)²)로 SSIM map 을 만든 뒤 평균낸다. scikit-image 없이 표준식·결정적.
    """
    c1, c2, k, s = (0.01 * 255) ** 2, (0.03 * 255) ** 2, (11, 11), 1.5
    mu_a, mu_b = cv2.GaussianBlur(gray_a, k, s), cv2.GaussianBlur(gray_b, k, s)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    s_a2 = cv2.GaussianBlur(gray_a * gray_a, k, s) - mu_a2
    s_b2 = cv2.GaussianBlur(gray_b * gray_b, k, s) - mu_b2
    s_ab = cv2.GaussianBlur(gray_a * gray_b, k, s) - mu_ab
    m = ((2 * mu_ab + c1) * (2 * s_ab + c2)) / ((mu_a2 + mu_b2 + c1) * (s_a2 + s_b2 + c2))
    return float(m.mean())


def hsv_correlation(jpeg_a: bytes, jpeg_b: bytes) -> float:
    """두 JPEG 의 HSV(H·S) 2D 히스토그램 상관계수(cv2.HISTCMP_CORREL, -1..1)(FR-303 보조).

    색감만 비슷한 다른 장면을 SSIM 애매 구간에서 보조 확인하는 용도(단독 skip 금지·FR-304).
    H(0~180)·S(0~256) 2채널 히스토그램을 normalize 후 상관계수로 비교한다.
    """
    img_a = cv2.imdecode(np.frombuffer(jpeg_a, np.uint8), cv2.IMREAD_COLOR)
    img_b = cv2.imdecode(np.frombuffer(jpeg_b, np.uint8), cv2.IMREAD_COLOR)
    if img_a is None or img_b is None:
        raise ValueError("hsv_correlation: JPEG 디코드 실패")
    hsv_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2HSV)
    hsv_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2HSV)
    hist_a = cv2.calcHist([hsv_a], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hist_b = cv2.calcHist([hsv_b], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist_a, hist_a, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist_b, hist_b, 0, 1, cv2.NORM_MINMAX)
    return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))


def _compare_indices(keep_count: int, config: KeyframeDedupConfig) -> range:
    """현재 프레임과 비교할 keep 인덱스 범위(FR-402). keep 리스트의 슬라이스 인덱스.

    - ``last``  : 마지막 1개만.
    - ``recent``: 최근 ``recent_window`` 개.
    - ``global`` 또는 기타(화이트리스트 밖): 전체(보수적 폴백 — 잘못된 모드로 과소 dedup 하지 않게,
      단 hist 단독 skip 은 어차피 금지라 안전).
    """
    if config.compare_mode == "last":
        n = 1
    elif config.compare_mode == "recent":
        n = max(1, config.recent_window)
    else:  # "global" → 전체 keep 비교(타임라인 손실 위험·비기본; 모드 검증은 config __post_init__).
        n = keep_count
    start = max(0, keep_count - n)
    return range(start, keep_count)


def dedup_keyframes(
    frames: list[KeyframeBytesResult],
    config: KeyframeDedupConfig,
) -> tuple[list[KeyframeBytesResult], list[dict]]:
    """키프레임 목록에서 near-dup 을 제거한다(FR-401~405·plan 알고리즘 verbatim).

    반환: ``(유지 프레임[원본 순서·메타 유지], skip 로그)``.
      - skip 로그 항목: ``{scene_index, reason, best_hamming, best_ssim}``
        (reason ∈ {``hash_candidate`` 미사용, ``ssim``, ``hist``}).
      - ``config.enabled=False`` 면 ``(frames, [])`` 그대로(완전 no-op·FR-103).

    알고리즘:
      1. ``scene_index`` 오름차순 처리. 프레임별 dHash·grayscale 1회 산출(재계산 방지).
      2. keep 비었으면 무조건 keep(FR-403).
      3. 비교집합 = mode 별(recent→최근 N · last→최근 1 · global→전체).
      4. 1차: 비교집합 중 Hamming ≤ hash_max 후보 없으면 keep(FR-202).
      5. 2차: 후보들과 max SSIM. ≥ ssim_min → skip(ssim).
         ∈[ssim_gray_lo, ssim_min) → HSV correlation max ≥ hist_min 면 skip(hist). 그 외 keep.
      6. 히스토그램 단독 skip 금지 — 반드시 4의 후보 통과 후에만(FR-304).
      7. 전부 skip 되어 keep 0 이면 마지막 처리 프레임 강제 keep(FR-404·C7).
    """
    if not config.enabled:
        return frames, []
    if not frames:
        return [], []

    # FR-401: scene_index 오름차순 처리(입력이 뒤섞여 있어도 결정적). 동률은 안정 정렬로 유지.
    ordered = sorted(frames, key=lambda f: f["scene_index"])

    keep: list[KeyframeBytesResult] = []
    keep_hash: list[int] = []
    keep_gray: list[np.ndarray] = []
    keep_jpeg: list[bytes] = []
    skips: list[dict] = []

    last_processed: KeyframeBytesResult | None = None
    for frame in ordered:
        last_processed = frame
        jpeg = frame["jpeg_bytes"]
        cur_hash = dhash(jpeg)

        # FR-403: keep 비었으면 무조건 keep.
        if not keep:
            keep.append(frame)
            keep_hash.append(cur_hash)
            keep_gray.append(_decode_gray(jpeg))
            keep_jpeg.append(jpeg)
            continue

        # 3·4: 비교집합 중 Hamming ≤ hash_max 후보 수집.
        cmp_range = _compare_indices(len(keep), config)
        candidates: list[int] = []
        best_hamming = None
        for i in cmp_range:
            d = hamming(cur_hash, keep_hash[i])
            if best_hamming is None or d < best_hamming:
                best_hamming = d
            if d <= config.hash_max:
                candidates.append(i)

        # FR-202: 후보 없으면 keep(1차 미통과 — SSIM/HSV 미적용·FR-304).
        if not candidates:
            keep.append(frame)
            keep_hash.append(cur_hash)
            keep_gray.append(_decode_gray(jpeg))
            keep_jpeg.append(jpeg)
            continue

        # 5: 후보들과 max SSIM(동일 resize 규격·FR-301).
        cur_gray = _decode_gray(jpeg)
        best_ssim = -1.0
        for i in candidates:
            sv = ssim(cur_gray, keep_gray[i])
            if sv > best_ssim:
                best_ssim = sv

        # 5a: max SSIM ≥ ssim_min → skip(ssim).
        if best_ssim >= config.ssim_min:
            skips.append(
                {
                    "scene_index": frame["scene_index"],
                    "reason": "ssim",
                    "best_hamming": int(best_hamming) if best_hamming is not None else None,
                    "best_ssim": round(best_ssim, 6),
                }
            )
            continue

        # 5b: 애매 구간 [ssim_gray_lo, ssim_min) → HSV correlation 보조(FR-303). 후보들 중 max.
        if config.ssim_gray_lo <= best_ssim < config.ssim_min:
            best_hist = -1.0
            for i in candidates:
                hv = hsv_correlation(jpeg, keep_jpeg[i])
                if hv > best_hist:
                    best_hist = hv
            if best_hist >= config.hist_min:
                skips.append(
                    {
                        "scene_index": frame["scene_index"],
                        "reason": "hist",
                        "best_hamming": int(best_hamming) if best_hamming is not None else None,
                        "best_ssim": round(best_ssim, 6),
                    }
                )
                continue

        # 그 외 keep(의미 차이 보존).
        keep.append(frame)
        keep_hash.append(cur_hash)
        keep_gray.append(cur_gray)
        keep_jpeg.append(jpeg)

    # FR-404·C7: 전부 skip 되어 keep 0 이면 마지막 처리 프레임 강제 keep(영상당 ≥1).
    # (FR-403 의 '첫 프레임 무조건 keep' 으로 keep≥1 이 이미 보장돼 현재는 도달하지 않는 안전망 —
    #  향후 비교 로직 변경 대비 유지한다.)
    if not keep and last_processed is not None:
        keep.append(last_processed)
        # 강제 keep 한 프레임은 skip 로그에서 제거(목록에 남지 않게).
        skips = [s for s in skips if s["scene_index"] != last_processed["scene_index"]]

    return keep, skips
