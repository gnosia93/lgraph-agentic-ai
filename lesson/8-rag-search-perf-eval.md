RAG(Retrieval-Augmented Generation) 시스템의 평가는 일반적인 LLM 평가와 완전히 다릅니다. 단순히 "답변이 그럴싸한가?"를 보는 게 아니라, **1) 문서를 찾아오는 '검색(Retrieval)'**과 **2) 찾아온 문서로 답변하는 '생성(Generation)'**의 두 가지 단계를 철저하게 분리해서 평가해야 합니다.
실무에서 가장 글로벌 표준으로 쓰이는 가이드라인은 RAG Triad(삼형제) 프레임워크입니다. 이를 기반으로 한 자동화 평가 도구(Ragas, DeepEval 등)를 활용해 점수화(‭$0 \sim 1$‬‭‬)하는 것이 정석입니다.

### 📐 1. RAG 평가의 핵심 기준: RAG Triad ###
RAG 파이프라인의 품질은 다음 세 가지 축으로 평가합니다.
```
       [ Query ]
        /     \
       /       \
  Context     Groundedness
 Relevance       /
     /          /
    v          v
[ Context ] ------> [ Response ]
           Answer
          Relevance
```
