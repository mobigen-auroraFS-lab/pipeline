"""도메인-불가지 분류 패키지 — per-asset 파이프라인의 classify 스테이지(route 직후·extract 전).

본체는 3-stage cascade(``cascade.classify``): Stage 1 시그니처 → 2 어휘 → 3 온프레미스 LLM.
도메인 지식은 코드 분기가 아니라 ``profiles.DomainProfile`` 데이터에 살고, ``domains/`` 도메인 팩이
import 부수효과로 ``DOMAIN_PROFILES`` 에 등록한다(새 도메인 = 프로파일 모듈 추가, cascade 코드 불변).
판정 결과(``types.ClassificationResult``)는 ``ingest.classification_persist`` 가 적재하며,
그 ``asset.domain_label`` 이 이후 도메인 팩 선택·deferred 판별의 분기점이 된다.
"""
