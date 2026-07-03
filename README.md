# LangGraph Agentic AI  
_This workshop is under construction, Not fully implemented yet_

본 워크샵은 EKS(Amazon Elastic Kubernetes Service) 환경에서 고성능 GPU 리소스를 효율적으로 관리하며, 단순한 RAG를 넘어 스스로 판단하고 실행하는 Agentic AI를 프로덕션 수준으로 구현하는 것을 목표로 합니다. 오픈 소스 LLM 모델 선정 및 파인튜닝과 더불어, 기업용 서비스의 필수 요건인 보안 가드레일(Guardrails), 정량적 평가(Evaluation), 그리고 지속적 통합/배포(CI/CD) 파이프라인을 포함하고 있으며, 인프라 구축부터 서비스 안정화 및 모니터링까지 LLM 서비스의 전체 생애주기(end-to-end) 에 대해 학습합니다.

완료 후 할 수 있는 것:     
- EKS에 GPU 노드그룹 구성 및 Karpenter 오토스케일
- LLM 파인튜닝과 정량 평가로 모델 품질 관리
- RAG + Agentic 워크플로우를 프로덕션급으로 배포
- 가드레일·관측성·LLMOps 파이프라인 구축

대상 독자: EKS 운영 경험이 있고 LLM 서비스를 실전에 도입하려는 엔지니어

### _Architecture_ ### 
![](https://github.com/gnosia93/eks-agentic-ai/blob/main/lesson/images/worshop-architecture.png)
_Drawn using Nano Banana 2.0 with a rough prompt, Not perfect but it's amazing !_
  
### _Topics_ ### 

* [L1. EKS 설치하기](https://github.com/gnosia93/eks-agentic-ai/tree/main/iac/tf)

* [L2. GPU 할당 및 주피터 설정](https://github.com/gnosia93/eks-agentic-ai/blob/main/lesson/2-pc-notebook.md)
     
* L3. LLM 선택하기
    - [일반 및 언어 모델링 능력 평가 (lm-eval-harness)](https://github.com/gnosia93/eks-agentic-ai/blob/main/lesson/3-llm-eval-harness.md) →  `이 모델이 기본은 되나?`
    - [도메인 적합성 (커스텀 벤치마크)](https://github.com/gnosia93/eks-agentic-ai/blob/main/lesson/3-llm-domain-capability.md) →  `우리 업무에 맞나?`
    - [응답 품질 (LLM-as-a-Judge)](https://github.com/gnosia93/eks-agentic-ai/blob/main/lesson/3-judge-as-a-llm.md) →  `실제 응답이 쓸만한가?`
    - [인퍼런스 벤치마킹](https://github.com/gnosia93/eks-agentic-ai/blob/main/lesson/3-inference-benchmarking.md) →  `빠른가? 비용효율적인가?`
      
* L4. RAG 파이프라인
   - [RAG 개요 (아키텍처와 동작 원리)](https://github.com/gnosia93/eks-agentic-ai/blob/main/lesson/5-rag-pipeline-concept.md)
   - [벡터DB 구축 (Milvus on EKS)](https://github.com/gnosia93/eks-agentic-ai/blob/main/lesson/5-milvus-install.md)
   - [문서 처리 파이프라인 (레이아웃 파싱 → 청킹 → 임베딩)](https://github.com/gnosia93/eks-agentic-ai/blob/main/lesson/5-pdf-save.md)
   - [RAG 백엔드 구현 (검색 → 리랭킹 → 답변 생성)](https://github.com/gnosia93/eks-agentic-ai/blob/main/lesson/5-llm-call.md)
   - [MCP 서버로 배포 (EKS)](https://github.com/gnosia93/eks-agentic-ai/blob/main/lesson/5-rag-mcp.md)   <--- test 필요
         
* [L5. LangGraph로 Agentic AI 만들기](https://github.com/gnosia93/langgraph-agentic-ai/blob/main/lesson/6-agentic-ai.md)
   - [1. 프로젝트 생성](https://github.com/gnosia93/lg-agentic-ai/blob/main/lesson/langgraph/6-project-init.md)
       
* [L6. 에이전트 가드레일 및 보안 강화](https://github.com/gnosia93/agentic-ai-eks/blob/main/lesson/7.guard-rail.md)
   - [입력값 필터링(PII 마스킹 & InputSanitizer)](https://github.com/gnosia93/eks-agentic-ai/blob/main/lesson/7-security-input-filter.md)
   - [Prompt Injection 탐지 모델 적용 (Llama Guard 3)](https://github.com/gnosia93/eks-agentic-ai/blob/main/lesson/7-prompt-inject.md)
   - [에이전트 도구 호출 보안: SSRF 방지를 위한 네트워크 격리 (Egress 관리)](https://github.com/gnosia93/eks-agentic-ai/blob/main/lesson/7-security-ssrf.md)
   - 권한 제어: LangGraph를 이용한 Human-in-the-loop(승인 절차) 구현
   - [NVIDIA NeMo Guardrails](https://github.com/gnosia93/langgraph-agentic-ai/blob/main/lesson/7-nemo-guardrails.md)
     
* L7. AI 시스템 평가 파이프라인
   - [RAG 검색 품질 평가](https://github.com/gnosia93/langgraph-agentic-ai/blob/main/lesson/8-rag-search-perf-eval.md)
   - [LLM 답변 품질 평가](https://github.com/gnosia93/langgraph-agentic-ai/blob/main/lesson/8-llm-response-eval.md)
   - [에이전트 행동 평가 (tool 선택, 실행 순서)](https://github.com/gnosia93/lgraph-agentic-ai/blob/main/lesson/8-agent-behaviour-eval.md)
   - [골든 데이터셋 구축](https://github.com/gnosia93/langgraph-agentic-ai/blob/main/lesson/8-golden-data.md)
     
    
* L8. LLMOps
   - [LLMOps의 개념](https://github.com/gnosia93/eks-agentic-ai/blob/main/lesson/9-cicd-llmops.md)
   - [프롬프트 버전 관리](https://github.com/gnosia93/eks-agentic-ai/blob/main/lesson/9-cicd-prompt-version.md)
   - [에이전트 로직 테스트 자동화](https://github.com/gnosia93/eks-agentic-ai/blob/main/lesson/9-cicd-agent-test.md)
   - 깃랩으로 통합하기
     
* L9. LLM 배포하기
   - [인퍼런스용 GPU 선정하기](https://github.com/gnosia93/eks-agentic-ai/blob/main/lesson/10-model-fit-gpu.md)
   - [vLLM 인퍼런스 (Qwen2.5-72B)](https://github.com/gnosia93/eks-agentic-ai/blob/main/lesson/10-vllm-inference.md)  
   - [TensorRT-LLM 인퍼런스 (Qwen2.5-72B)](https://github.com/gnosia93/eks-agentic-ai/blob/main/lesson/10-tensorrt-inference.md)
   - [LLM 추론 성능(inference performance) 비교](https://github.com/gnosia93/eks-agentic-ai/blob/main/lesson/10-llm-benchmark.md)
   - [KEDA 기반 오토 스케일링](https://github.com/gnosia93/eks-agentic-ai/blob/main/lesson/10-keda-autoscaling.md)
        
* L10. 멀티노드 인퍼런스 구성
   - [LeaderWorkerSet](https://github.com/gnosia93/eks-agentic-ai/blob/main/lesson/11-LeaderWorkerSet.md)
   - [KubeRay](https://github.com/gnosia93/langgraph-agentic-ai/blob/main/lesson/11-rayserve.md)
     
* L11. [서빙 최적화](https://github.com/gnosia93/langgraph-agentic-ai/blob/main/lesson/12-inference-optimization.md)
   - [Prefix Aware Routing](https://github.com/gnosia93/langgraph-agentic-ai/blob/main/lesson/12-prefix-aware-routing.md)
   - [Model routing / cascading](https://github.com/gnosia93/langgraph-agentic-ai/blob/main/lesson/12-model-routing.md)
      - [bifrost 게이트웨이](https://github.com/gnosia93/langgraph-agentic-ai/blob/main/lesson/12-bifrost-fallback.md)
      - [LiteLLM](https://github.com/gnosia93/langgraph-agentic-ai/blob/main/lesson/12-litellm.md)
  

* L12. Observability
   - [LangFuse 사용하기](https://github.com/gnosia93/langgraph-agentic-ai/blob/main/lesson/13-langfuse.md)

* [L13. Post Training](https://github.com/gnosia93/eks-agentic-ai/blob/main/lesson/14-post-training.md)
   - [정제 모델 생성하기(100B to 1B Distillation)](https://github.com/gnosia93/lgraph-agentic-ai/blob/main/lesson/14-post-training-distillation.md)
   - [파인튜닝하기-인스트럭션 튜닝](https://github.com/gnosia93/agentic-ai-eks/blob/main/lesson/4-finetune.md)
     
### _Appendix_ ###

* [Prometheus / Loki Stack](https://github.com/gnosia93/eks-agentic-ai/blob/main/lesson/a2-prometheus-loki.md)   
* [AI Engineering - Prompt/Context/Harness](https://github.com/gnosia93/eks-agentic-ai/blob/main/lesson/a5.harness-eng.md)
* [promptfoo로 모델 평가하기](https://github.com/gnosia93/eks-agentic-ai/blob/main/lesson/a7.promptfoo.md)
* [promptfoo로 모델 취약점 점검하기](https://github.com/gnosia93/langgraph-agentic-ai/blob/main/lesson/a7.promptfoo-security.md)
* [TensorRT-LLM Speculative 디코딩](https://github.com/gnosia93/eks-agentic-ai/blob/main/lesson/a8-speculative-decoding.md)
* [Neo4j로 지식 그래프 구현하기 (온톨로지)](https://github.com/gnosia93/eks-agentic-ai/blob/main/lesson/a9-ontology-neo4j.md)      
* Agentic AI
  - [멀티 에이전트 오케스트레이션](https://github.com/gnosia93/eks-agentic-ai/blob/main/lesson/6-multi-agent.md)
  - [프로덕션 최적화 및 고려사항](https://github.com/gnosia93/eks-agentic-ai/blob/main/lesson/6-production-optim.md)
  - 사용자 피드백 루프: 응답에 대한 thumbs up/down 수집하고 개선에 반영하는 구조.
* [Reasoning Model 모델](https://github.com/gnosia93/langgraph-agentic-ai/blob/main/lesson/a10-reasoning.md)


## 레퍼런스 ##

* [Stanford CME295 Transformers & LLMs | Autumn 2025](https://www.youtube.com/watch?v=Ub3GoFaUcds&list=PLoROMvodv4rOCXd21gf0CF4xr35yINeOy)


