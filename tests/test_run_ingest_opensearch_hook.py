"""run_ingest OpenSearch 증분 훅 단위 테스트 (spec 020 G3 — OS·DB 불필요).

이 그룹은 **프로덕션 적재(run_ingest) 경로 변경**이라 안전 게이트가 최우선이다:

  (a) ``OPENSEARCH_SYNC_ENABLED`` 미설정(기본) → 훅 헬퍼가 **즉시 반환**한다. opensearch_sync /
      opensearch-py 를 **import 도 하지 않는다**(지연 import). 따라서 미도입 환경의 run_ingest 가
      완전 불변(SC-001, 회귀 0). sys.modules 에 센티넬을 꽂아 그 모듈의 어떤 속성도 접근되지 않음을 단언.
  (b) enabled + 색인 실패(예외 주입) → 헬퍼가 예외를 **삼키고**(warning 로그) **정상 반환**한다.
      OS 가 죽어도 ingest 는 중단·롤백되지 않는다(SC-003, 격리).
  (c) enabled + 정상 → ``index_asset`` 이 해당 asset_id 로 호출된다(증분 색인 정상 경로, FR-002).

모든 검증은 **가짜 settings·db·opensearch_sync 모듈 주입**으로 OS·DB 없이 수행한다 — 실제 색인은 G5.
"""

from __future__ import annotations

import contextlib
import sys
import types
import unittest
import uuid
from unittest import mock

from processing.app import run_ingest as ri
from processing.ingest import pipeline_steps as ps  # FR-E3: 스텝 정본(seam patch)

_ASSET = uuid.UUID("018f0000-0000-7000-8000-000000000019")


def _settings(*, enabled: bool, channel: str = "st") -> types.SimpleNamespace:
    """run_ingest 가 cfg 로 쓰는 설정 대역(필요 필드만)."""
    return types.SimpleNamespace(
        opensearch=types.SimpleNamespace(
            sync_enabled=enabled,
            url="http://localhost:9200",
            index="assets",
            filename_noise_patterns=(),
        ),
        embed=types.SimpleNamespace(active_channel=channel),
    )


class _RecordingModule(types.ModuleType):
    """sys.modules 에 꽂는 가짜 opensearch_sync 모듈 — 어떤 속성이든 접근되면 기록한다.

    훅이 ``from src.search.opensearch_sync import ...`` 를 실행하면(=import) 속성 접근이 일어나
    ``accessed`` 에 남는다. off 경로에서 이 리스트가 비어 있으면 **모듈을 import 조차 안 한 것**(SC-001).
    """

    def __init__(self) -> None:
        super().__init__("src.search.opensearch_sync")
        self.accessed: list[str] = []
        self.index_asset = mock.MagicMock(name="index_asset")
        self.get_client = mock.MagicMock(name="get_client", return_value=mock.MagicMock())

    def __getattribute__(self, name: str):
        # 내부 속성(accessed/index_asset/get_client/dunder)은 기록 대상에서 제외, 그 외 접근만 기록.
        if not name.startswith("_") and name not in {"accessed", "index_asset", "get_client"}:
            object.__getattribute__(self, "accessed").append(name)
        return object.__getattribute__(self, name)


@contextlib.contextmanager
def _patched_os_sync():
    """가짜 opensearch_sync 모듈을 sys.modules 에 주입(지연 import 가 이걸 잡는다)."""
    fake = _RecordingModule()
    with mock.patch.dict(sys.modules, {"src.search.opensearch_sync": fake}):
        yield fake


class TestOpenSearchIndexerFactory(unittest.TestCase):
    """배치 색인기 ``_make_opensearch_indexer`` 단위 검증 — 안전 게이트 + 클라이언트 1회 재사용.

    팩토리가 돌려준 콜러블 ``index(asset_id)`` 를 자산 finalize 직후마다 부른다. off 안전·격리·
    채널/인덱스 전달은 종전 헬퍼와 동일하게 보장하되, 클라이언트는 배치당 1회만 생성한다.
    """

    def setUp(self) -> None:
        self.db = mock.MagicMock()

    def _indexer(self, settings):
        return ri._make_opensearch_indexer(db=self.db, settings=settings)

    # (a) off → 콜러블 즉시 반환 + opensearch_sync 미접근(미import)
    def test_disabled_returns_immediately_without_touching_module(self) -> None:
        with _patched_os_sync() as fake:
            self._indexer(_settings(enabled=False))(_ASSET)
        # index/client 미호출
        fake.index_asset.assert_not_called()
        fake.get_client.assert_not_called()
        # 모듈 속성 접근이 전혀 없었음 = from ... import 자체가 실행되지 않음(지연 import 보장)
        self.assertEqual(fake.accessed, [])
        # DB 트랜잭션도 열지 않음(읽기 conn 미오픈)
        self.db.transaction.assert_not_called()

    # (a') PR4b·③: 방어적 getattr 폐지 — 색인기는 완전한 PipelineSettings(.opensearch 보유)를
    #      요구한다. malformed settings(.opensearch 없음)는 조용히 off 가 아니라 팩토리에서
    #      즉시 AttributeError(fail-fast — 오설정이 런타임까지 숨지 않게).
    def test_malformed_settings_without_opensearch_raises(self) -> None:
        with self.assertRaises(AttributeError):
            self._indexer(object())

    # (b) enabled + 색인 예외 → 삼킴(warning) + 무중단 반환
    def test_enabled_index_error_is_swallowed_with_warning(self) -> None:
        with _patched_os_sync() as fake:
            fake.index_asset.side_effect = RuntimeError("OS down")
            with self.assertLogs(ri._LOG, level="WARNING") as log:
                # 예외가 전파되면 이 호출이 raise → 테스트 실패. 격리 성공이면 정상 반환.
                ret = self._indexer(_settings(enabled=True))(_ASSET)
        self.assertIsNone(ret)
        self.assertTrue(any("asset_id" in m or "opensearch" in m.lower() for m in log.output))

    # (b') enabled + get_client 단계 예외(OS 미도달)도 삼킨다 — 적재 무중단
    def test_enabled_client_error_is_swallowed(self) -> None:
        with _patched_os_sync() as fake:
            fake.get_client.side_effect = ConnectionError("unreachable")
            with self.assertLogs(ri._LOG, level="WARNING"):
                ret = self._indexer(_settings(enabled=True))(_ASSET)
        self.assertIsNone(ret)
        fake.index_asset.assert_not_called()

    # (c) enabled + 정상 → index_asset 이 해당 asset_id 로 호출
    def test_enabled_indexes_with_asset_id(self) -> None:
        with _patched_os_sync() as fake:
            self._indexer(_settings(enabled=True))(_ASSET)
        fake.index_asset.assert_called_once()
        call = fake.index_asset.call_args
        # asset_id 가 (문자열로) 전달되었는지 — 위치/키워드 어디에 있든 인자 집합에서 확인
        passed = list(call.args) + list(call.kwargs.values())
        self.assertIn(str(_ASSET), [str(a) for a in passed])
        # 색인 대상 채널/인덱스가 settings 에서 전달됨
        self.assertEqual(call.kwargs.get("index"), "assets")
        self.assertEqual(call.kwargs.get("channel"), "st")
        # 읽기 conn 을 열었음(PG 읽기전용 → OS 쓰기)
        self.db.transaction.assert_called_once()

    # (d) 배치 다건 → 클라이언트는 1회만 생성·재사용(연결 셋업 절감), 색인·읽기 conn 은 자산마다
    def test_client_created_once_and_reused_across_batch(self) -> None:
        with _patched_os_sync() as fake:
            index = self._indexer(_settings(enabled=True))
            for i in range(3):
                index(uuid.UUID(int=i))
        fake.get_client.assert_called_once()  # 배치당 1회 생성(자산마다 새 연결 X)
        self.assertEqual(fake.index_asset.call_count, 3)  # 자산마다 색인
        self.assertEqual(self.db.transaction.call_count, 3)  # 자산마다 별도 읽기 트랜잭션


# ── (a) run_ingest 전체 경로에서 off → OS 코드 미접촉(회귀 0) ──────────────────


def _route(modality="txt", routable=True, reason="", domain="general"):
    from processing.ingest.router import RouteResult

    return RouteResult("/d/a." + modality, modality, domain, routable, reason)


def _patch_ingest(stack: contextlib.ExitStack) -> dict:
    """run_ingest 의존 함수 전부 patch(실 DB·LLM·파일 불필요) — test_run_ingest 와 동형."""
    from processing.dispatch.types import AssetRecord

    specs = {
        "route_file": mock.patch.object(ps, "route_file", return_value=_route()),
        "file_hash_and_size": mock.patch.object(ps, "file_hash_and_size", return_value=("h0", 10)),
        "find_registered_asset_by_hash": mock.patch.object(
            ps, "find_registered_asset_by_hash", return_value=None
        ),
        "create_asset": mock.patch.object(ps, "create_asset", return_value=1),
        "record_classification": mock.patch.object(ps, "record_classification"),
        "set_status": mock.patch.object(ps, "set_status"),
        "finalize_asset": mock.patch.object(ps, "finalize_asset"),
        "mark_failed": mock.patch.object(ri, "mark_failed"),
        "record_lineage": mock.patch.object(ps, "record_lineage"),
        "validate_ext_meta": mock.patch.object(ps, "validate_ext_meta"),
    }
    m = {k: stack.enter_context(p) for k, p in specs.items()}
    m["_AssetRecord"] = AssetRecord
    return m


class TestRunIngestOffPathUnchanged(unittest.TestCase):
    """off(기본) 에서 run_ingest 가 OpenSearch 모듈을 전혀 건드리지 않음을 전체 경로로 단언."""

    def test_success_path_does_not_touch_opensearch_when_off(self) -> None:
        db = mock.MagicMock()
        with contextlib.ExitStack() as stack:
            m = _patch_ingest(stack)
            with _patched_os_sync() as fake:
                res = ri.run_ingest(
                    ["/d/a.txt"],
                    db=db,
                    extract_fn=lambda ctx: m["_AssetRecord"](),
                    classify_fn=lambda p, mod: __import__(
                        "processing.classify.types", fromlist=["ClassificationResult"]
                    ).ClassificationResult(final_label="general", confidence=0.7, decided_stage=2),
                    settings=_settings(enabled=False),
                )
        # 적재 자체는 성공(기존 동작 불변)
        self.assertEqual(res["registered"], [1])
        # OpenSearch 모듈은 import 조차 안 됨(속성 접근 0 · 색인 0)
        self.assertEqual(fake.accessed, [])
        fake.index_asset.assert_not_called()
        fake.get_client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
