## 추론 최적화 기술 모음 ##

### 1. 배치·스케줄링

* Continuous batching (in-flight batching): 요청을 정적 배치가 아니라 토큰 단위로 계속 끼워넣어 GPU를 꽉 채움. 처리량 핵심. (vLLM의 기본)
* Chunked prefill: 긴 prefill을 잘게 쪼개 decode와 섞어, 긴 프롬프트가 다른 요청을 막지 않게 함.
* Prefill/Decode 분리(Disaggregation): prefill(연산 집약)과 decode(메모리 집약)를 다른 GPU/풀로 분리. llm-d, Dynamo의 핵심.

### 2. 메모리·KV 캐시

* PagedAttention: KV를 블록 단위로 관리(단편화 제거). (이미 다룸)
* KV 캐시 양자화: KV를 FP8/INT8로 저장해 메모리 절감.
* KV offloading / 계층화: 안 쓰는 KV를 CPU RAM·NVMe로 내림 (LMCache 등).
* Prefix/Prompt Caching: 공통 prefix 재사용. (이미 다룸)

### 3. 모델 경량화·연산 최적화

* 양자화(Quantization): INT8/FP8/INT4 (GPTQ, AWQ 등)로 모델을 줄여 속도·메모리 개선.
* FlashAttention: 어텐션을 메모리 효율적으로 계산하는 커널.
* Speculative Decoding(추측 디코딩): 작은 초안 모델이 여러 토큰을 미리 뽑고 큰 모델이 한 번에 검증 → 지연 단축. (Medusa, EAGLE, n-gram 등)
* 병렬화: Tensor / Pipeline / Expert(MoE) Parallelism — 큰 모델을 여러 GPU에 분산.

### 4. 라우팅·오토스케일 (시스템 레벨)

* Prefix-aware / KV-aware Routing: (이미 다룸)
* Model routing / cascading: 쉬운 건 작은 모델, 어려운 건 큰 모델로(비용 최적화).
* Load-aware routing: 큐 길이·KV 잔여량 보고 분배.
* 오토스케일링: 요청량에 따라 GPU 파드 확장/축소 (KEDA, queue 기반).

### 5. 출력·디코딩 최적화

* Structured / constrained decoding: JSON·문법 강제(스키마 준수)로 재시도 줄임.
* Streaming (TTFT 개선): 첫 토큰부터 흘려보내 체감 지연 단축.

### 6. 측정·운영 지표

* 핵심 지표: TTFT(첫 토큰 시간), TBT/ITL(토큰 간 지연), Throughput(tok/s), GPU 활용률, 비용/1K토큰.
* 관측·SLO: 지연 분포(p50/p99), 큐 대기, 캐시 적중률 모니터링.
