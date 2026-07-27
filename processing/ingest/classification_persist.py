"""F-5.1 분류 결과 영속화.

``asset_classification`` 적재 + ``asset.domain_label``/``domain_confidence`` 갱신.
final_label == 'review' 는 별도 큐 없이 ``asset.domain_label='review'`` 표식만 남긴다
(검토 대기 큐 테이블은 없다 — 사람 검토 흐름은 아직 정하지 않았다).
psycopg ``Connection`` 을 받아 오케스트레이터 트랜잭션에서 조합한다.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from psycopg import Connection

from processing.classify.types import ClassificationResult
from src.database.ids import uuid7


def record_classification(conn: Connection[Any], asset_id: uuid.UUID, result: ClassificationResult) -> None:
    """분류 결과를 asset_classification 에 적재하고 asset 도메인 라벨을 갱신한다.

    재분류(동일 asset_id 재실행) 시 ON CONFLICT 로 최신 결과로 덮어쓴다.
    stage*_scores 는 cascade 각 단계(시그니처/제로샷/LLM)가 산출한 도메인별 점수 jsonb 이며,
    ``decided_stage`` 가 실제로 최종 판정을 내린 단계다. 이후 팩 선택·deferred 판별의 기반.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO asset_classification (
                classification_id, asset_id, stage1_scores, stage2_scores, stage3_scores,
                final_label, confidence, decided_stage, policy_version
            ) VALUES (%s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s, %s, %s, %s)
            ON CONFLICT (asset_id) DO UPDATE SET
                stage1_scores = EXCLUDED.stage1_scores,
                stage2_scores = EXCLUDED.stage2_scores,
                stage3_scores = EXCLUDED.stage3_scores,
                final_label   = EXCLUDED.final_label,
                confidence    = EXCLUDED.confidence,
                decided_stage = EXCLUDED.decided_stage,
                policy_version = EXCLUDED.policy_version
            """,
            (
                uuid7(), asset_id,
                json.dumps(result.stage1_scores, ensure_ascii=False),
                json.dumps(result.stage2_scores, ensure_ascii=False),
                json.dumps(result.stage3_scores, ensure_ascii=False),
                result.final_label, result.confidence, result.decided_stage, result.policy_version,
            ),
        )
        # asset 테이블도 동기화 — run_ingest 가 domain_label 로 팩 선택·deferred 판별.
        # ⚠️ '검토 필요' 판정도 이 표식이 전부다 — 존재하지 않는 큐 테이블에 넣으려 하면 안 된다.
        cur.execute(
            "UPDATE asset SET domain_label = %s, domain_confidence = %s, updated_at = now() WHERE asset_id = %s",
            (result.final_label, result.confidence, asset_id),
        )
