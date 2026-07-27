"""영상 키프레임 중복 제거 코어 — 무거운 모델을 돌리기 전에 결정적으로 2단계 판정한다.

JPEG 추출 직후·VLM 이전에 시각적으로 거의 동일한 키프레임을 결정적 2단계로 걸러낸다:

  1) **1차 dHash(perceptual hash)** — 64-bit. 비교 keep 집합 중 Hamming ≤ ``hash_max`` 면 near-dup
     후보(빠른 CPU 필터).
  2) **2차 SSIM/HSV** — 후보만 동일 resize 규격으로 SSIM. ``≥ ssim_min`` 이면 중복 확정(skip).
     SSIM ∈ [``ssim_gray_lo``, ``ssim_min``) 애매 구간에서만 HSV histogram correlation 보조 판정
     (``≥ hist_min`` 이면 버린다). ⚠️ **색 분포만으로는 절대 버리지 않는다** — 반드시 1차 후보를
     통과한 것에만 보조로 쓴다(색이 비슷해도 내용이 다른 장면이 흔하다).

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

# 기본값은 여기 하나뿐 — 설정 폴백과 공유하는 경량 상수 모듈(무거운 영상 라이브러리 무의존).
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

    설정에서 영상 skill 이 만들어 주입한다 — 임계를 두 곳에 두면 한쪽만 바뀐다.
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
        """설정 값이 허용 범위인지 확인한다 — 잘못된 값은 생성 시점에 막는다."""
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
    """두 해시가 몇 비트 다른지 센다(가장 싼 비교 — 정수 연산 두 번).

    Args:
        a: 첫 해시.
        b: 둘째 해시.

    Returns:
        다른 비트 수. 작을수록 비슷하다.
    """
    return bin(a ^ b).count("1")


def dhash(jpeg_bytes: bytes) -> int:
    """JPEG 를 64비트 지각 해시로 바꾼다 — 비슷한 그림이 비슷한 해시를 갖는다.

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
    """두 그림의 구조적 유사도를 낸다(0~1 · 1에 가까울수록 같다).

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
    """두 그림의 색 분포가 얼마나 닮았는지 낸다(-1~1).

    구조 비교가 애매한 구간에서만 보조로 쓴다. ⚠️ **이것만으로 버리면 안 된다**.
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
    """현재 프레임을 어느 범위의 남긴 프레임들과 비교할지 정한다.

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
    """거의 같은 키프레임을 걸러 낸다 — 무거운 시각 모델을 돌리기 **전에**.

    같은 장면이 여러 번 잡히면 그만큼 모델을 헛돌린다. 판정은 **싼 것부터 두 단계**로 한다:
    먼저 해시로 비슷한 후보만 골라 내고(비교가 거의 무료), 그 후보에만 실제 화면 비교를
    돌린다. 화면 비교를 전부에 돌리면 걸러 내는 비용이 아끼는 비용을 넘는다.

    ⚠️ **색 분포만으로는 버리지 않는다** — 색이 비슷해도 내용이 다른 장면이 흔하다. 반드시
    해시 후보를 통과한 것에만 보조 판정으로 쓴다.

    ⚠️ **결과가 0장이 되지 않게 한다** — 전부 비슷하다고 판정되면 마지막 한 장은 남긴다.
    한 장도 없으면 그 영상은 검색에서 아예 사라진다.

    Args:
        frames: 장면별 키프레임. **입력이 뒤섞여 있어도** 장면 순으로 정렬해 처리하므로
            결과가 흔들리지 않는다.
        config: 임계·비교 범위 설정. **꺼져 있으면 입력을 그대로** 돌려준다(완전 무동작).

    Returns:
        ``(남긴 프레임[원래 순서·메타 유지], 버린 이유 로그)``. 로그에는 무엇과 얼마나
        비슷해서 버렸는지가 담겨, 임계를 조정할 근거가 된다.
    """
    if not config.enabled:
        return frames, []
    if not frames:
        return [], []

    # 장면 순으로 처리한다 — 입력이 뒤섞여 있어도 결과가 같아야 한다(동률은 들어온 순서 유지).
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

        # 첫 프레임은 비교 대상이 없으니 무조건 남긴다.
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

        # 해시로 비슷한 후보가 없으면 남긴다 — 비싼 화면 비교를 아예 돌리지 않는다.
        if not candidates:
            keep.append(frame)
            keep_hash.append(cur_hash)
            keep_gray.append(_decode_gray(jpeg))
            keep_jpeg.append(jpeg)
            continue

        # 후보들과 구조 비교 — 가장 비슷한 값을 쓴다.
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

        # 애매한 구간에서만 색 분포를 보조로 본다(후보들 중 최댓값).
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

    # ⚠️ 한 장도 안 남으면 그 영상이 검색에서 통째로 사라진다 — 마지막 프레임을 강제로 남긴다.
    # (첫 프레임을 무조건 남기므로 현재 구조에서는 도달하지 않는 안전망 —
    #  향후 비교 로직 변경 대비 유지한다.)
    if not keep and last_processed is not None:
        keep.append(last_processed)
        # 강제 keep 한 프레임은 skip 로그에서 제거(목록에 남지 않게).
        skips = [s for s in skips if s["scene_index"] != last_processed["scene_index"]]

    return keep, skips
