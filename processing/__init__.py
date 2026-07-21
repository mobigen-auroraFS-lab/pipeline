"""처리 파이프라인 패키지 — 077/078 레포 분리로 DataFlatform ``src.*`` 에서 이관.

코어(``src.*``: config·database·llm·embedders·search·relations·topic·registry·domain·file)는
설치/참조하고, 파이프라인 자기 코드는 ``processing.*`` top-level 로 둔다(설치 코어 src.* 와 충돌 회피).
"""
