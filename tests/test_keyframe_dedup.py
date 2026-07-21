"""048 G1 — 영상 키프레임 near-dup 제거 순수 코어 단위 테스트.

``src/preprocess/keyframe_dedup.py`` 의 dHash·Hamming·SSIM·HSV correlation·orchestration
(``dedup_keyframes``)을 합성 JPEG(단색·미세 변형·재등장)로 검증한다. LLM·DB·실파일 0(순수 단위).
결정성(헌법 3조)·임계 경계·FR-2xx/3xx/4xx/6xx·SC-002~005 를 덮는다.
"""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

import cv2
import numpy as np

from processing.preprocess.keyframe_dedup import (
    HASH_SIZE,
    SSIM_SIZE,
    KeyframeDedupConfig,
    dedup_keyframes,
    dhash,
    hamming,
    hsv_correlation,
    ssim,
)


def _jpeg(color: tuple[int, int, int], size: int = 128) -> bytes:
    """단색 BGR JPEG bytes. color 는 (B, G, R)."""
    img = np.full((size, size, 3), color, dtype=np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


def _grad(size: int = 128) -> bytes:
    """좌→우 밝기 그라디언트 JPEG(전혀 다른 장면)."""
    row = np.linspace(0, 255, size, dtype=np.uint8)
    img = np.tile(row, (size, 1))
    return cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))[1].tobytes()


def _grad_v(size: int = 128) -> bytes:
    """상→하 밝기 그라디언트(다른 장면 — 가로 그라디언트와도 다름)."""
    col = np.linspace(0, 255, size, dtype=np.uint8).reshape(size, 1)
    img = np.tile(col, (1, size))
    return cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))[1].tobytes()


def _frame(idx: int, jpeg: bytes) -> dict:
    """KeyframeBytesResult 최소형(scene_index 오름차순 처리 검증용 메타 포함)."""
    return {
        "scene_index": idx,
        "start_sec": float(idx),
        "end_sec": idx + 1.0,
        "frame_sec": idx + 0.5,
        "jpeg_bytes": jpeg,
    }


class TestModuleSkeleton(unittest.TestCase):
    """T101: 상수·dataclass 가 plan Interfaces 와 일치하는지."""

    def test_constants(self) -> None:
        self.assertEqual(HASH_SIZE, 8)
        self.assertEqual(SSIM_SIZE, 128)

    def test_config_defaults(self) -> None:
        cfg = KeyframeDedupConfig(enabled=True)
        self.assertEqual(cfg.hash_max, 7)
        self.assertEqual(cfg.ssim_min, 0.94)
        self.assertEqual(cfg.ssim_gray_lo, 0.90)
        self.assertEqual(cfg.hist_min, 0.97)
        self.assertEqual(cfg.compare_mode, "recent")
        self.assertEqual(cfg.recent_window, 4)

    def test_config_is_frozen(self) -> None:
        # 결정성(헌법 3조): 동일 설정 객체는 불변 — 런타임 변이 차단.
        cfg = KeyframeDedupConfig(enabled=True)
        with self.assertRaises(FrozenInstanceError):
            cfg.enabled = False  # type: ignore[misc]


class TestConfigValidation(unittest.TestCase):
    """권고5/6: 오설정 fail-fast — ``__post_init__`` 가 잘못된 모드·범위를 ValueError 로 차단."""

    def test_invalid_compare_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            KeyframeDedupConfig(enabled=True, compare_mode="bogus")

    def test_negative_hash_max_raises(self) -> None:
        with self.assertRaises(ValueError):
            KeyframeDedupConfig(enabled=True, hash_max=-1)

    def test_ssim_out_of_range_raises(self) -> None:
        with self.assertRaises(ValueError):
            KeyframeDedupConfig(enabled=True, ssim_min=1.5)

    def test_gray_lo_above_min_raises(self) -> None:
        with self.assertRaises(ValueError):
            KeyframeDedupConfig(enabled=True, ssim_gray_lo=0.95, ssim_min=0.94)

    def test_recent_window_below_one_raises(self) -> None:
        with self.assertRaises(ValueError):
            KeyframeDedupConfig(enabled=True, recent_window=0)


class TestHamming(unittest.TestCase):
    """T102 (FR-203): 정수 XOR + popcount."""

    def test_known_distance(self) -> None:
        self.assertEqual(hamming(0b1011, 0b1110), 2)

    def test_zero_for_equal(self) -> None:
        self.assertEqual(hamming(5, 5), 0)


class TestDhash(unittest.TestCase):
    """T103 (FR-201): 동일 bytes→hamming 0, 단색 vs 그라디언트→큰 거리."""

    def test_identical_bytes_same_hash(self) -> None:
        j = _jpeg((100, 100, 100))
        self.assertEqual(hamming(dhash(j), dhash(j)), 0)

    def test_solid_vs_gradient_large_distance(self) -> None:
        self.assertGreater(hamming(dhash(_jpeg((100, 100, 100))), dhash(_grad())), 10)

    def test_decode_failure_raises(self) -> None:
        with self.assertRaises(ValueError):
            dhash(b"not a jpeg")


class TestSsim(unittest.TestCase):
    """T104 (FR-301): 동일 gray≈1.0, 단색 vs 그라디언트<0.9."""

    def test_identical_near_one(self) -> None:
        from processing.preprocess.keyframe_dedup import _decode_gray

        g = _decode_gray(_jpeg((128, 128, 128)))
        self.assertGreaterEqual(ssim(g, g), 0.999)

    def test_different_below_threshold(self) -> None:
        from processing.preprocess.keyframe_dedup import _decode_gray

        a = _decode_gray(_jpeg((20, 20, 20)))
        b = _decode_gray(_grad())
        self.assertLess(ssim(a, b), 0.9)


class TestHsvCorrelation(unittest.TestCase):
    """T105 (FR-303): 동일 단색≈1.0, 색상 크게 다른 두 단색은 낮음."""

    def test_identical_near_one(self) -> None:
        j = _jpeg((30, 60, 200))
        self.assertGreaterEqual(hsv_correlation(j, j), 0.999)

    def test_different_colors_low(self) -> None:
        # 빨강(BGR=(0,0,255)) vs 파랑(BGR=(255,0,0)) — HSV hue 가 크게 다름.
        red = _jpeg((0, 0, 255))
        blue = _jpeg((255, 0, 0))
        self.assertLess(hsv_correlation(red, blue), 0.5)


class TestDedupKeyframes(unittest.TestCase):
    """T106: dedup_keyframes orchestration(FR-102~104·202~304·402~404·SC-002~005)."""

    def test_disabled_noop(self) -> None:
        # FR-103: enabled=False → (frames, []) 그대로(동일 객체·순서).
        frames = [_frame(1, _jpeg((10, 10, 10))), _frame(2, _jpeg((10, 10, 10)))]
        kept, skips = dedup_keyframes(frames, KeyframeDedupConfig(enabled=False))
        self.assertIs(kept, frames)
        self.assertEqual(skips, [])

    def test_burst_keep_one(self) -> None:
        # SC-002: 동일 단색 10장 burst → keep 1.
        j = _jpeg((100, 100, 100))
        frames = [_frame(i, j) for i in range(1, 11)]
        kept, skips = dedup_keyframes(frames, KeyframeDedupConfig(enabled=True))
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["scene_index"], 1)
        self.assertEqual(len(skips), 9)

    def test_progressive_burst(self) -> None:
        # SC-003: A→B(≈A)→C(≈B) 점진 burst → keep [A] only (recent·last).
        # 미세 밝기 차이라 dHash 후보·SSIM≥0.94 로 연쇄 skip 되어야 한다.
        a, b, c = _jpeg((40, 40, 40)), _jpeg((42, 42, 42)), _jpeg((44, 44, 44))
        frames = [_frame(1, a), _frame(2, b), _frame(3, c)]
        for mode in ("recent", "last"):
            kept, _ = dedup_keyframes(
                frames, KeyframeDedupConfig(enabled=True, compare_mode=mode)
            )
            self.assertEqual([k["scene_index"] for k in kept], [1], f"mode={mode}")

    def test_recent_window_reappear(self) -> None:
        # SC-004: A → (서로 다른 장면 ≥4) → A'(=A bytes).
        #   recent(N=4): A 가 비교창 밖 → A' keep(둘 다 유지).
        #   global: 전체 비교 → A' skip(A 와 동일).
        a = _jpeg((10, 10, 10))
        fillers = [_grad(), _grad_v(), _jpeg((200, 50, 50)), _jpeg((50, 200, 50)), _jpeg((50, 50, 200))]
        seq = [a, *fillers, a]
        frames = [_frame(i + 1, j) for i, j in enumerate(seq)]

        kept_recent, _ = dedup_keyframes(
            frames, KeyframeDedupConfig(enabled=True, compare_mode="recent", recent_window=4)
        )
        recent_idx = [k["scene_index"] for k in kept_recent]
        self.assertIn(1, recent_idx)               # 첫 A
        self.assertIn(len(seq), recent_idx)         # 재등장 A'(마지막) — 둘 다 keep

        kept_global, skips_global = dedup_keyframes(
            frames, KeyframeDedupConfig(enabled=True, compare_mode="global")
        )
        global_idx = [k["scene_index"] for k in kept_global]
        self.assertIn(1, global_idx)
        self.assertNotIn(len(seq), global_idx)      # global 에선 A' skip
        self.assertTrue(any(s["scene_index"] == len(seq) for s in skips_global))

    def test_all_skip_keeps_one(self) -> None:
        # SC-005·FR-404: 전부 동일이어도 최종 keep ≥ 1.
        j = _jpeg((77, 77, 77))
        frames = [_frame(i, j) for i in range(1, 6)]
        kept, _ = dedup_keyframes(frames, KeyframeDedupConfig(enabled=True))
        self.assertGreaterEqual(len(kept), 1)

    def test_keep_order_and_meta(self) -> None:
        # FR-102: keep 프레임의 scene_index·start_sec 원본 유지·시간순.
        a, b = _jpeg((10, 10, 10)), _grad()
        # 입력을 일부러 뒤섞어 넣어도 scene_index 오름차순 처리(FR-401)·메타 보존 확인.
        frames = [_frame(3, b), _frame(1, a), _frame(2, _jpeg((10, 10, 10)))]
        kept, _ = dedup_keyframes(frames, KeyframeDedupConfig(enabled=True))
        idxs = [k["scene_index"] for k in kept]
        self.assertEqual(idxs, sorted(idxs))  # 시간순
        # 메타 보존: scene_index 1 의 start_sec 은 1.0
        f1 = next(k for k in kept if k["scene_index"] == 1)
        self.assertEqual(f1["start_sec"], 1.0)
        self.assertEqual(f1["frame_sec"], 1.5)

    def test_hist_only_never_skips(self) -> None:
        # FR-304: hash 후보 미통과(그라디언트 vs 단색)면 SSIM/HSV 미적용·keep.
        # 두 프레임은 dHash 거리가 커 후보가 안 되므로 둘 다 keep 되어야 한다.
        frames = [_frame(1, _grad()), _frame(2, _jpeg((100, 100, 100)))]
        kept, skips = dedup_keyframes(frames, KeyframeDedupConfig(enabled=True))
        self.assertEqual(len(kept), 2)
        # hist 사유 단독 skip 이 없어야 한다.
        self.assertFalse(any(s.get("reason") == "hist" for s in skips))

    def test_dhash_collision_ssim_keeps(self) -> None:
        # FR-602·FR-301: 단색 두 장은 dHash 가 충돌(둘 다 0·hamming≤hash_max → 후보)하지만,
        # 밝기가 크게 달라 SSIM < ssim_min → 2차 SSIM 이 dHash 오탐을 구제해 둘 다 keep(임계 밖 keep).
        dark, bright = _jpeg((40, 40, 40)), _jpeg((210, 210, 210))
        self.assertLessEqual(hamming(dhash(dark), dhash(bright)), 7)  # 전제: dHash 후보 성립
        frames = [_frame(1, dark), _frame(2, bright)]
        kept, skips = dedup_keyframes(frames, KeyframeDedupConfig(enabled=True))
        self.assertEqual([k["scene_index"] for k in kept], [1, 2])
        self.assertEqual(skips, [])

    def test_empty_input(self) -> None:
        kept, skips = dedup_keyframes([], KeyframeDedupConfig(enabled=True))
        self.assertEqual(kept, [])
        self.assertEqual(skips, [])

    def test_skip_log_fields(self) -> None:
        # FR-405: skip 로그에 scene_index·reason·best_hamming·best_ssim 포함.
        j = _jpeg((100, 100, 100))
        frames = [_frame(1, j), _frame(2, j)]
        _, skips = dedup_keyframes(frames, KeyframeDedupConfig(enabled=True))
        self.assertEqual(len(skips), 1)
        log = skips[0]
        for key in ("scene_index", "reason", "best_hamming", "best_ssim"):
            self.assertIn(key, log)
        self.assertEqual(log["scene_index"], 2)
        self.assertEqual(log["reason"], "ssim")


class TestDeterminism(unittest.TestCase):
    """T107 (헌법 3조): 동일 입력 2회 → keep·skip 완전 동일."""

    def test_repeated_runs_identical(self) -> None:
        a = _jpeg((10, 10, 10))
        seq = [a, _grad(), _grad_v(), _jpeg((42, 42, 42)), _jpeg((200, 50, 50)), a]
        frames = [_frame(i + 1, j) for i, j in enumerate(seq)]
        cfg = KeyframeDedupConfig(enabled=True)
        k1, s1 = dedup_keyframes(frames, cfg)
        k2, s2 = dedup_keyframes(frames, cfg)
        self.assertEqual([k["scene_index"] for k in k1], [k["scene_index"] for k in k2])
        self.assertEqual(s1, s2)


if __name__ == "__main__":
    unittest.main()
