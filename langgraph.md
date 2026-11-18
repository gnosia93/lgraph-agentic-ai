# LangGraph 학습 커리큘럼 (6주)

> 대상: Python과 LLM API 호출에 익숙하고 LangGraph는 처음인 엔지니어
> 기간: 6주, 주당 6~8시간 (개념 2~3시간, 실습 4~5시간)
> 기준 버전: LangGraph 1.2.x, LangChain 1.3 이상 (2026년 11월)
> 결과물: 주차별 실습 코드, 테스트·평가·배포까지 마친 캡스톤 프로젝트 1개

## 학습 목표

6주가 끝나면 다음을 할 수 있어야 합니다.

1. 에이전트 루프(think → execute → observe)를 LangGraph로 직접 구현하고, `create_agent`가 내부에서 무엇을 하는지 설명한다.
2. 체크포인터로 실행 상태를 저장해서 사람 승인, 장애 후 재개, 과거 시점부터 다시 실행하기를 구현한다.
3. 라우팅, 병렬 처리(map-reduce), 서브그래프, 멀티에이전트 중 문제에 맞는 구조를 고른다.
4. 에이전트를 테스트·평가·추적해서 품질을 숫자로 보여 주고 배포한다.

## 로드맵

| 주차 | 주제 | 핵심 API | 산출물 |
|---|---|---|---|
| 0 | 환경 준비 | | 개발 환경, 모델 연결 확인 |
| 1 | 그래프 기초 | `StateGraph`, reducer, 조건부 엣지 | 문의 분류 워크플로 |
| 2 | 에이전트 루프 | `MessagesState`, `bind_tools`, `create_agent`, middleware | 직접 만든 ReAct 에이전트 |
| 3 | 영속성과 사람 개입 | checkpointer, `interrupt`, `Command(resume=...)`, time travel | 승인이 필요한 환불 에이전트 |
| 4 | 메모리와 스트리밍 | Store, 메시지 정리, stream mode, runtime context | 사용자를 기억하는 상담 봇 |
| 5 | 복잡한 흐름 | `Send`, `Command(goto=...)`, 서브그래프, 멀티에이전트 | 병렬 리서치 에이전트 |
| 6 | 프로덕션 | 테스트, 평가, 추적, 배포 | 캡스톤 완성 |

---

## 0주차. 환경 준비 (2시간)

### 사전 지식 점검

- Python 타입 힌트: `TypedDict`, `Annotated`, `Literal`
- `async`/`await` 기초 (스트리밍과 서버 배포에서 사용)
- LLM tool calling 개념: 모델은 도구 이름과 인자를 내놓기만 하고, 실행은 코드가 한다

### 설치

```bash
uv init langgraph-study && cd langgraph-study
uv add langgraph langchain "langgraph-cli[inmem]" langgraph-checkpoint-sqlite python-dotenv
uv add langchain-openai      # 사용할 모델 제공자 패키지 (langchain-anthropic, langchain-aws 등)
```

### 모델 선택

tool calling을 지원하는 모델이면 무엇이든 됩니다.

| 선택지 | 장점 | 권장 시점 |
|---|---|---|
| 호스팅 API (OpenAI, Anthropic, Amazon Bedrock 등) | tool calling이 안정적이라 개념 학습에 집중하기 좋다 | 1~5주차 |
| 직접 띄운 vLLM (예: Qwen3-8B) | 비용을 직접 통제하고 서빙까지 연습할 수 있다 | 6주차에 교체해 비교 |

vLLM을 쓸 때는 tool calling 옵션을 켜고 OpenAI 호환 클라이언트로 연결합니다.

```bash
vllm serve Qwen/Qwen3-8B --enable-auto-tool-choice --tool-call-parser hermes
```

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    base_url="http://localhost:8000/v1", api_key="EMPTY", model="Qwen/Qwen3-8B",
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},   # Qwen3 thinking 모드 끄기
)
```

### 추적 도구 (하나 고르기)

- **LangSmith**: LangGraph와 기본으로 연동되고 Studio와 함께 쓰기 좋다. 환경 변수 `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY` 설정
- **MLflow**: 직접 호스팅한다. `mlflow.langchain.autolog()` 한 줄이면 노드, 도구, LLM 호출이 span으로 남는다 ([문서](https://mlflow.org/docs/latest/tracing/integrations/langgraph))
- **W&B Weave**: `weave.init("프로젝트명")` 한 줄이면 LangChain 연동으로 모델과 도구 호출이 자동 추적된다 ([문서](https://docs.coreweave.com/products/wandb/weave/guides/integrations/langchain))

---

## 1주차. 그래프 기초: State, Node, Edge

### 목표

- 노드들이 공유 상태(State)를 주고받으며 그래프가 실행되는 방식을 이해한다.
- 일반 함수 단계와 LLM 단계를 한 그래프에 섞어 쓴다.

### 핵심 개념

- **State**: 그래프 전체가 공유하는 데이터(`TypedDict`). 노드는 바뀐 키만 dict로 반환한다.
- **Reducer**: 같은 키에 업데이트가 여러 개 들어올 때 합치는 규칙. 리스트는 `Annotated[list, operator.add]`, 메시지는 `add_messages`를 쓴다.
- **Edge**: 고정 순서(`add_edge`), 조건 분기(`add_conditional_edges`), 루프
- `compile()`, `invoke()`, 그래프 시각화(`get_graph().draw_mermaid()`)

### 읽을거리

- [Thinking in the Graph API](https://docs.langchain.com/oss/python/langgraph/thinking-in-graph-api)
- [Graph API overview](https://docs.langchain.com/oss/python/langgraph/graph-api), [Define and update state](https://docs.langchain.com/oss/python/langgraph/graph-api-state), [Sequences, branches, and loops](https://docs.langchain.com/oss/python/langgraph/graph-api-control-flow)
- [Workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)의 prompt chaining, routing, parallelization 부분
- LangChain Academy [Quickstart: LangGraph Essentials](https://academy.langchain.com/courses/langgraph-essentials-python)

### 실습: 고객 문의 분류 워크플로

1. state에 `text`, `category`, `reply`를 둔다. `classify` 노드는 처음에는 키워드 규칙으로 billing/tech/other를 분류한다(LLM 없이 시작).
2. 조건부 엣지로 카테고리별 답변 노드 3개로 분기한다.
3. `classify`를 LLM structured output(`model.with_structured_output(...)`)으로 바꾼다.
4. 감정 분석 노드와 긴급도 판단 노드를 병렬로 추가하고, 둘 다 같은 키(`notes`)에 결과를 쓰게 한다. reducer가 없을 때 `InvalidUpdateError`가 나는 것을 확인한 뒤 `Annotated[list[str], operator.add]`로 고친다.
5. 그래프를 Mermaid로 그려 README에 넣는다.

### 확인 질문

- 노드가 state 전체가 아니라 바뀐 키만 반환하는 이유는 무엇인가?
- reducer가 없는 키에 병렬 노드 두 개가 동시에 쓰면 어떻게 되는가?
- 조건부 엣지의 라우팅 함수는 무엇을 반환하는가?

---

## 2주차. 에이전트 루프 직접 만들기

### 목표

- think → execute → observe 루프를 노드 2개와 조건부 엣지로 구현한다.
- LangChain `create_agent`가 같은 루프를 감싼 것임을 확인하고, middleware로 루프 중간에 끼어드는 법을 익힌다.

### 핵심 개념

- `MessagesState`와 `add_messages` reducer (메시지 ID 기준으로 추가하거나 교체)
- `model.bind_tools(tools)`, `AIMessage.tool_calls`, `ToolMessage(tool_call_id=...)`
- 종료 조건: tool_calls가 없으면 END로 간다. 무한 루프는 `recursion_limit`으로 막는다.
- `create_agent`와 middleware hook: `before_model`, `after_model`, `wrap_model_call`, `wrap_tool_call`

### 읽을거리

- [LangGraph quickstart](https://docs.langchain.com/oss/python/langgraph/quickstart)
- [LangChain agents](https://docs.langchain.com/oss/python/langchain/agents), [Custom middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom)

### 실습

1. 아래 뼈대로 주문 조회 도구 2~3개를 가진 에이전트를 만든다.

```python
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain.messages import ToolMessage
from langchain.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph

@tool
def get_order_status(order_id: str) -> str:
    """주문 상태를 조회한다."""
    return {"A100": "배송 중", "A200": "환불 완료"}.get(order_id, "없는 주문")

tools = [get_order_status]
tools_by_name = {t.name: t for t in tools}
model = init_chat_model("<provider>:<model>").bind_tools(tools)   # 0주차에서 고른 모델

def think(state: MessagesState):                  # think: 다음 행동 결정
    return {"messages": [model.invoke(state["messages"])]}

def execute(state: MessagesState):                # execute + observe: 실행 결과를 메시지로 추가
    results = []
    for call in state["messages"][-1].tool_calls:
        output = tools_by_name[call["name"]].invoke(call["args"])
        results.append(ToolMessage(content=str(output), tool_call_id=call["id"]))
    return {"messages": results}

def should_continue(state: MessagesState) -> Literal["execute", END]:
    return "execute" if state["messages"][-1].tool_calls else END

builder = StateGraph(MessagesState)
builder.add_node("think", think)
builder.add_node("execute", execute)
builder.add_edge(START, "think")
builder.add_conditional_edges("think", should_continue)
builder.add_edge("execute", "think")
agent = builder.compile()

agent.invoke({"messages": [{"role": "user", "content": "A100 주문 어디쯤이야?"}]})
```

2. 도구가 예외를 던지면 에러 내용을 ToolMessage로 돌려줘서 모델이 인자를 고쳐 다시 호출하게 한다.
3. 같은 기능을 `create_agent(model, tools)`로 다시 만들고, 추적 화면에서 두 실행의 단계를 비교한다.
4. `wrap_tool_call` middleware를 붙여 도구 이름, 인자, 실행 시간을 기록한다.
5. `recursion_limit`을 작게 설정해 `GraphRecursionError`를 일부러 내 본다.

### 확인 질문

- think, execute, observe 각 단계는 위 코드의 어디에 해당하는가?
- ToolMessage에 `tool_call_id`가 필요한 이유는 무엇인가?
- 직접 구현과 `create_agent`는 각각 언제 쓰는가?

---

## 3주차. 영속성, 사람 개입, 장애 대응

### 목표

- 체크포인터로 단계마다 상태를 저장하고, 같은 thread로 이어서 실행한다.
- 실행 중간에 멈춰 사람 승인을 받은 뒤 재개한다.
- 실패한 단계만 다시 실행하고, 과거 시점에서 갈라져 다른 경로로 실행한다.

### 핵심 개념

- checkpointer(`InMemorySaver`, `SqliteSaver`, `PostgresSaver`)와 `thread_id`
- `interrupt()`로 멈추고 `Command(resume=값)`으로 재개한다. 재개하면 그 노드는 처음부터 다시 실행된다.
- time travel: `get_state_history()`, `update_state()`, 특정 체크포인트부터 다시 실행
- 노드별 retry, timeout, 에러 처리

### 읽을거리

- [Persistence](https://docs.langchain.com/oss/python/langgraph/persistence), [Checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers)
- [Interrupts](https://docs.langchain.com/oss/python/langgraph/interrupts), [Use time-travel](https://docs.langchain.com/oss/python/langgraph/use-time-travel)
- [Fault tolerance](https://docs.langchain.com/oss/python/langgraph/fault-tolerance), [Configure nodes](https://docs.langchain.com/oss/python/langgraph/graph-api-nodes)

### 실습: 승인이 필요한 환불 에이전트

1. 2주차 에이전트에 `refund(order_id, amount)` 도구를 추가한다.
2. 환불 호출이 나오면 승인 노드로 보내고, `interrupt()`로 멈춘 뒤 승인, 거절, 금액 수정 중 하나로 재개한다.

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt

def approve_refund(state: MessagesState):
    call = state["messages"][-1].tool_calls[0]
    decision = interrupt({"action": "refund", "args": call["args"]})   # 여기서 멈춘다
    ...  # decision에 따라 실행, 거절, 인자 수정

agent = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "ticket-42"}}
agent.invoke({"messages": [...]}, config)          # interrupt에서 멈추고 반환
agent.invoke(Command(resume="approve"), config)    # 같은 thread로 재개
```

3. `SqliteSaver`로 바꾼다. 승인 대기 중에 프로세스를 종료했다가 다시 띄우고, 같은 `thread_id`로 재개되는지 확인한다.
4. 일부러 실패하는 도구에 retry policy를 건다. 실패 후 재실행할 때 앞 단계가 다시 실행되지 않는지 확인한다.
5. 실행 기록에서 환불 직전 시점을 골라 다른 금액으로 갈라 실행해 본다.

### 확인 질문

- `interrupt()`보다 앞에 둔 코드(예: 외부 API 호출)는 재개할 때 어떻게 되는가? 그렇다면 어떻게 설계해야 하는가?
- `thread_id`를 빠뜨리면 무슨 일이 생기는가?
- 체크포인터가 있으면 장애 복구가 왜 쉬워지는가?

---

## 4주차. 메모리, 스트리밍, 컨텍스트

### 목표

- 대화 안의 기억(thread)과 대화를 넘나드는 기억(Store)을 구분해서 구현한다.
- 긴 대화에서 메시지를 줄이거나 요약해 컨텍스트 길이를 관리한다.
- 토큰과 진행 상황을 실시간으로 보여 준다.

### 핵심 개념

- 단기 메모리는 checkpointer의 thread 상태, 장기 메모리는 Store(`InMemoryStore`, `PostgresStore`)에 둔다. Store는 namespace 단위로 저장하고 검색한다.
- 메시지 정리: 자르기(trim), 요약, `RemoveMessage`
- stream mode: `values`, `updates`, `messages`(토큰), `custom`(`get_stream_writer()`), 이벤트 스트리밍
- runtime context(`context_schema`): 사용자 ID나 DB 연결처럼 실행마다 고정된 값을 state와 따로 전달

### 읽을거리

- [Memory](https://docs.langchain.com/oss/python/langgraph/add-memory), [Stores](https://docs.langchain.com/oss/python/langgraph/stores), [Long-term memory](https://docs.langchain.com/oss/python/langchain/long-term-memory)
- [Streaming](https://docs.langchain.com/oss/python/langgraph/streaming), [Event streaming](https://docs.langchain.com/oss/python/langgraph/event-streaming)

### 실습

1. 사용자가 말한 선호(언어, 배송지 등)를 Store에 저장하고, 새 thread에서 불러와 답변에 반영한다.
2. 메시지가 N개를 넘으면 오래된 메시지를 요약 메시지 하나로 바꾸는 노드를 추가한다.
3. CLI에서 `stream_mode=["messages", "updates"]`로 토큰과 노드 진행 상황을 함께 출력한다.
4. 오래 걸리는 도구 안에서 진행률을 `custom` 스트림으로 내보낸다.

### 확인 질문

- 어떤 정보를 state에 둘지, Store에 둘지, runtime context에 둘지 어떻게 정하는가?
- 요약으로 메시지를 줄이면 무엇을 잃는가?
- `updates` 스트림과 `values` 스트림은 어떻게 다른가?

---

## 5주차. 복잡한 흐름: 병렬, 서브그래프, 멀티에이전트

### 목표

- 실행 중에 개수가 정해지는 병렬 작업을 `Send`로 처리한다.
- 큰 그래프를 서브그래프로 나누고, 여러 멀티에이전트 구조 중 문제에 맞는 것을 고른다.

### 핵심 개념

- `Send`: map-reduce (질문을 N개로 쪼개 병렬로 실행한 뒤 합치기)
- `Command(goto=..., update=...)`: 노드 안에서 상태 변경과 다음 노드 이동을 함께 처리
- 서브그래프: 컴파일한 그래프를 노드로 쓰기, 부모와 상태 키를 공유하거나 변환하기
- 멀티에이전트 패턴: router, subagents(supervisor), handoffs, skills
- (선택) Functional API: `@entrypoint`, `@task`

### 읽을거리

- [Use Send and Command](https://docs.langchain.com/oss/python/langgraph/graph-api-send-command), [Subgraphs](https://docs.langchain.com/oss/python/langgraph/use-subgraphs)
- [Workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents)의 orchestrator-worker, evaluator-optimizer 부분
- [Multi-agent](https://docs.langchain.com/oss/python/langchain/multi-agent), [Functional API](https://docs.langchain.com/oss/python/langgraph/functional-api)

### 실습: 병렬 리서치 에이전트

1. planner 노드가 질문을 하위 질문 3~5개로 쪼갠다.
2. `Send`로 하위 질문마다 검색 서브그래프를 병렬로 실행한다.
3. writer 노드가 결과를 합쳐 보고서를 쓰고, reviewer 노드가 기준에 못 미치면 writer로 되돌린다(최대 2회).
4. 같은 문제를 supervisor(subagents) 방식으로 다시 만들고, 추적 화면에서 LLM 호출 수, 토큰, 지연 시간을 비교한다.

### 확인 질문

- 조건부 엣지 대신 `Send`가 필요한 경우는 언제인가?
- `Command(goto=...)`와 조건부 엣지는 각각 언제 쓰는가?
- 멀티에이전트로 나누면 늘어나는 비용은 무엇이고, 단일 에이전트가 더 나은 경우는 언제인가?

---

## 6주차. 프로덕션: 테스트, 평가, 추적, 배포

### 목표

- LLM 없이 도는 단위 테스트와 실제 모델로 하는 평가를 나눠 만든다.
- 에이전트를 서버로 띄우고 배포한다.

### 핵심 개념

- 단위 테스트: 가짜 채팅 모델과 `InMemorySaver`로 노드와 라우팅을 검증한다.
- 평가: 최종 답변과 경로(trajectory, 어떤 도구를 어떤 순서로 불렀는지)를 함께 평가한다. 정답 비교와 LLM-as-judge를 섞어 쓴다.
- 앱 구조와 `langgraph.json`, `langgraph dev`로 띄우는 로컬 서버와 Studio
- 배포 경로 두 가지
  - LangSmith Agent Server로 배포
  - 직접 운영: FastAPI로 그래프 감싸기, `PostgresSaver`/`PostgresStore`, 컨테이너 이미지, Kubernetes(EKS) 배포. 요청에서 `thread_id`를 받아 config로 넘기고, 스트리밍은 SSE로 보낸다.
- 그래프를 바꿔 배포할 때 실행 중인 thread와의 호환성

### 읽을거리

- [Test](https://docs.langchain.com/oss/python/langgraph/test), [Unit testing](https://docs.langchain.com/oss/python/langchain/test/unit-testing), [Agent evals](https://docs.langchain.com/oss/python/langchain/test/evals)
- [Application structure](https://docs.langchain.com/oss/python/langgraph/application-structure), [Run a local server](https://docs.langchain.com/oss/python/langgraph/local-server), [LangSmith Studio](https://docs.langchain.com/oss/python/langgraph/studio)
- [Deployment](https://docs.langchain.com/oss/python/langgraph/deploy), [Backward compatibility](https://docs.langchain.com/oss/python/langgraph/backward-compatibility), [Observability](https://docs.langchain.com/oss/python/langgraph/observability)

### 실습

아래 캡스톤 프로젝트를 마무리한다.

### 확인 질문

- 단위 테스트로 잡을 문제와 평가로 잡을 문제는 각각 무엇인가?
- 서버를 여러 대로 늘리면 체크포인터를 무엇으로 바꿔야 하는가?
- 배포 후 그래프 구조를 바꾸면 승인 대기 중이던 thread는 어떻게 되는가?

---

## 캡스톤: 고객 지원 티켓 에이전트 (5~6주차)

1~4주차 실습을 하나로 합칩니다.

```text
티켓 입력
  │
  ▼
[입력 검사] 개인정보, 프롬프트 인젝션 → 위험하면 사람에게 넘김
  │
  ▼
[라우터] billing / tech / other
  ├── billing 에이전트: 주문 조회, 환불 (사람 승인 필요)
  ├── tech 에이전트: 문서 검색(RAG), 장애 조회
  └── other: 일반 답변
  │
  ▼
[검토] 답변 품질과 개인정보 노출 확인 → 응답
```

### 요구사항

1. Postgres 체크포인터를 써서, 승인 대기 중에 서버를 재시작해도 이어서 진행된다.
2. 고객별 선호와 이전 문의 요약을 Store에 저장해 다음 문의에 활용한다.
3. 답변 토큰과 처리 단계를 스트리밍한다.
4. 티켓 30~50개로 평가셋을 만들고 라우팅 정확도, 도구 경로 정확도, 답변 품질, 평균 지연 시간, 티켓당 비용을 잰다.
5. 추적(LangSmith, MLflow, Weave 중 하나)을 붙이고, 로컬 서버 또는 EKS에 배포한다.

### 확장 과제

- 라우터를 LLM structured output, 작은 분류 모델(Decision Model 등), 규칙 기반으로 바꿔 가며 정확도, 지연 시간, 비용을 표로 비교한다.
- 모델을 직접 띄운 vLLM으로 바꾸고 호스팅 API와 품질, 비용을 비교한다.
- 참고 예제: W&B의 [LangGraph와 Weave로 만드는 고객 지원 HITL 튜토리얼](https://forge.coreweave.com/wandb/ai-team-articles/Human-in-the-loop%20AI%20for%20customer%20service%20with%20LangGraph%20and%20W&B%20Weave/reports/Tutorial-Human-in-the-loop-AI-for-customer-service-with-LangGraph-and-W-B-Weave--VmlldzoxNzA0MjQyMQ)

### 완료 기준

- README에 아키텍처 그림, 실행 방법, 평가 결과 표, 주요 설계 결정과 그 이유가 있다.
- 처음 보는 사람이 README만 보고 실행할 수 있다.

---

## 부록 A. 자주 막히는 지점

| 증상 | 원인 | 해결 |
|---|---|---|
| 메시지가 쌓이지 않고 덮어써짐 | messages 키에 reducer가 없음 | `MessagesState` 또는 `Annotated[list, add_messages]` 사용 |
| 병렬 노드에서 `InvalidUpdateError` | reducer 없는 키에 동시에 씀 | 해당 키에 reducer 지정 |
| `GraphRecursionError` | 종료 조건에 도달하지 않음 | 라우팅 함수 확인, 반복 횟수를 state로 관리 |
| interrupt 재개 후 외부 API가 두 번 호출됨 | 재개하면 노드가 처음부터 다시 실행됨 | 부작용이 있는 코드는 interrupt 뒤나 별도 노드로 옮김 |
| 다시 실행했는데 이전 대화가 없음 | `thread_id` 누락이나 변경, 체크포인터 없음 | `config={"configurable": {"thread_id": ...}}` 확인 |
| 체크포인트 저장 오류 | state에 직렬화할 수 없는 객체(DB 연결 등)가 들어감 | state에는 데이터만 두고, 연결 객체는 runtime context로 전달 |
| 직접 띄운 vLLM 모델이 도구를 안 부름 | tool calling 옵션 누락 | `--enable-auto-tool-choice --tool-call-parser hermes` |

## 부록 B. 다음 단계

- [Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview): 계획, 하위 에이전트, 파일 시스템 도구를 갖춘 LangGraph 기반 에이전트 하네스
- [MCP](https://docs.langchain.com/oss/python/langchain/mcp): 외부 MCP 서버의 도구를 에이전트에 연결
- LangChain Academy: [Deep Research with LangGraph](https://academy.langchain.com/courses/deep-research-with-langgraph), [Ambient Agents with LangGraph](https://academy.langchain.com/courses/ambient-agents), [전체 강좌 목록](https://academy.langchain.com/collections)
