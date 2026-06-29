## Prefix Aware Routing (KV 캐시 인지 라우팅 / 서빙 최적화) ##

vLLM 앞단 라우팅 서버는 크게 세 부류예요: ① vLLM 진영 공식, ② Kubernetes 표준(Gateway API), ③ 범용 LLM 게이트웨이.

### ① vLLM 진영 (가장 직접적) ###

* vLLM Production Stack: vLLM 팀 공식 K8s 배포 스택. prefix-aware routing + KV 캐시 공유(LMCache 연동) 내장. "같은 prompt prefix 요청을 같은 인스턴스로" 보내줘요. (문서) — vLLM만 쓴다면 1순위 후보.
* vLLM Router: 위 스택의 라우터 컴포넌트. round-robin 말고 prefill/decode 인지 + 세션 어피니티 + prefix 인지 같은 정교한 알고리즘 제공. (소개)

### ② Kubernetes 표준 — Gateway API Inference Extension (GIE) ###

* llm-d + Endpoint Picker(EPP): K8s 네이티브 분산 추론 프레임워크. EPP가 KV/prefix 캐시 인지 라우팅을 담당. Red Hat·Google·NVIDIA·IBM 등이 미는 표준화 흐름이고 CNCF로 들어갔어요. (가이드)
* GKE Inference Gateway: 위 EPP 기반의 구글 매니지드 버전(cache-aware routing). GKE 한정이지만 개념 참고용. (설명)
이 방향의 장점: Kubernetes Gateway API 표준을 따르므로 EKS에도 이식성이 좋아요.

### ③ 기타 K8s 네이티브 라우터 ###

* AIBrix (ByteDance): vLLM 대규모 서빙용. prefix-cache-aware 라우팅 등 포함.
* Kthena Router (Volcano 진영): K8s 네이티브, model-aware + 실시간 메트릭 기반 라우팅. (블로그)
* SGLang Router: SGLang을 쓸 경우 자체 cache-aware 라우터(RadixAttention과 궁합).
* NVIDIA Dynamo: GPU 추론용 스마트 라우터(KV-aware) 포함.

### ④ 범용 LLM 게이트웨이 (prefix 인지는 약하거나 별도 설정) ###

* LiteLLM Proxy: 여러 모델·벤더 통합, 로드밸런싱·키 관리·관측. 다만 KV/prefix-aware는 핵심이 아니에요(주로 모델 라우팅·페일오버용).
* Envoy / NGINX + consistent hashing: 직접 구성. 세션/해시 기반 sticky routing 정도는 DIY 가능하지만 prefix 캐시 상태까지는 못 봐요.



vLLM 앞단 라우터의 본질 기능은 "prefix/KV 캐시를 가진 인스턴스로 보내기". 가장 표준적인 길은 ① vLLM Production Stack(간편) 또는 ② Gateway API Inference Extension(표준)이에요.
