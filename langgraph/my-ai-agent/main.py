import json
from typing import Annotated, AsyncGenerator, Literal
from typing_extensions import TypedDict

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# ==========================================
# 1. 도구(Tool) 정의
# ==========================================
@tool
def get_weather(city: str) -> str:
    """특정 도시의 현재 날씨를 조회합니다."""
    # 실제 서비스에서는 날씨 API를 호출하면 됩니다.
    return f"{city}의 현재 날씨는 맑고 기온은 18도입니다."

tools = [get_weather]

# ==========================================
# 2. vLLM (Qwen3-8B) 모델 연결
# ==========================================
llm = ChatOpenAI(
    model="Qwen/Qwen3-8B",
    openai_api_key="EMPTY",  # vLLM 로컬 서버는 아무 문자열이나 입력
    openai_api_base="http://localhost:8000/v1",  # vLLM 서버 주소
    temperature=0.7,
    streaming=True,  # ⭐ SSE 스트리밍을 위해 반드시 True 설정!
).bind_tools(tools)

# ==========================================
# 3. LangGraph 상태(State) 및 그래프 구축
# ==========================================
class State(TypedDict):
    messages: Annotated[list, add_messages]

async def agent_node(state: State):
    # 비동기 호출(ainvoke)을 사용해야 이벤트 스트림이 막힘없이 흐릅니다.
    response = await llm.ainvoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "__end__"

# 그래프 조립
builder = StateGraph(State)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue)
builder.add_edge("tools", "agent")

# 대화 내역(메모리) 유지를 위한 체크포인터 연결
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# ==========================================
# 4. FastAPI 웹서버 및 SSE 스트리밍 설정
# ==========================================
app = FastAPI()

# static 폴더 마운트
app.mount("/static", StaticFiles(directory="static"), name="static")

class ChatRequest(BaseModel):
    message: str
    thread_id: str = "user_session_1"

async def event_generator(message: str, thread_id: str) -> AsyncGenerator[str, None]:
    inputs = {"messages": [("user", message)]}
    config = {"configurable": {"thread_id": thread_id}}

    try:
        async for event in graph.astream_events(inputs, config=config, version="v2"):
            kind = event["event"]

            # LLM 답변 토큰 실시간 스트리밍
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    payload = json.dumps({"type": "token", "content": chunk.content}, ensure_ascii=False)
                    yield f"data: {payload}\n\n"

            # 도구 호출 시작 알림
            elif kind == "on_tool_start":
                payload = json.dumps({"type": "tool_start", "tool": event["name"]}, ensure_ascii=False)
                yield f"data: {payload}\n\n"

            # 도구 호출 완료 알림
            elif kind == "on_tool_end":
                payload = json.dumps({"type": "tool_end", "tool": event["name"]}, ensure_ascii=False)
                yield f"data: {payload}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except Exception as e:
        error_payload = json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False)
        yield f"data: {error_payload}\n\n"

# 메인 웹페이지 반환
@app.get("/")
async def index():
    return FileResponse("static/index.html")

# 채팅 SSE 스트리밍 API
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    return StreamingResponse(
        event_generator(req.message, req.thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
