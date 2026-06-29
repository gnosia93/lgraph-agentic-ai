
Model routing은 "어느 모델로 보낼까" 라는 축이라, 그 후보 모델이 Bedrock 모델이든 셀프호스팅 오픈소스 모델이든 다 대상이 돼요. 오히려 그게 model routing의 대표적인 실무 시나리오예요.

### Model routing 종류 ###

* 품질/난이도 기반 (cascading): 쉬운 요청은 작고 싼 모델, 어려우면 크고 비싼 모델로 단계적 승급. (예: Llama-8B로 먼저 → 부족하면 Claude로 에스컬레이션)
* 벤더/호스팅 혼합 (이게 질문하신 부분): Bedrock 매니지드 모델 ↔ 자체 EKS의 OSS 모델(vLLM) 사이 라우팅.
* 작업 유형 기반: 코딩은 모델 A, 요약은 모델 B, 한국어는 모델 C 식 특화 라우팅.
* 비용/지연/가용성 기반: 싼 모델 우선, 한도 초과·장애 시 폴백(failover).

### Bedrock ↔ OSS 혼합 라우팅, 왜 하나 (전형적 패턴) ###

* 비용 최적화: 대량의 쉬운 트래픽은 EKS의 OSS 모델(고정비)로, 어렵거나 중요한 건 Bedrock 고성능 모델로.
* 민감도 분리: 민감 데이터는 자체 호스팅 OSS로(데이터 유출 우려↓), 일반 요청은 Bedrock으로.
* 폴백/가용성: 한쪽이 한도 초과·장애면 다른 쪽으로 우회.
* 품질 캐스케이드: OSS로 1차 시도 → confidence 낮으면 Bedrock으로 재시도.
구분 주의 (중요)


### 구현 방식 ###

* LiteLLM Proxy: Bedrock·OpenAI·자체 vLLM 등을 하나의 OpenAI 호환 엔드포인트로 묶고, 라우팅·폴백·로드밸런싱·비용추적. 벤더 혼합 라우팅의 대표 도구.
* 라우터 모델/분류기: 요청을 보고 "쉬움/어려움"을 판단해 모델을 고르는 작은 분류기 (RouteLLM 같은 연구·OSS).
* 게이트웨이류: Portkey, Kong AI Gateway, 자체 게이트웨이 등.
* AWS 맥락: Bedrock 자체에도 요청을 적절한 모델로 보내는 Intelligent Prompt Routing 기능이 있어요(주로 Bedrock 내 모델군 사이). Bedrock ↔ 외부 OSS까지 묶으려면 보통 LiteLLM 같은 상위 게이트웨이를 둬요.

