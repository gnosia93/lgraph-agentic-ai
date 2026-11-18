
### 폴더 구조 및 사전 준비 ###
 
```
my-ai-agent/
├── main.py              # FastAPI 서버 + LangGraph 에이전트 전체 코드
└── static/
    └── index.html       # 브라우저에서 열릴 채팅 화면 UI
```

```bash
pip install fastapi uvicorn langchain-openai langgraph
```

### 실행 방법 ###

```
vllm serve Qwen/Qwen3-8B --enable-auto-tool-choice --tool-call-parser hermes --port 8000
python main.py
```

웹 브라우저를 열고 http://localhost:8080 으로 접속해 *"서울 날씨 어때?"*라고 입력해 보시면, 도구 실행 알림이 뜬 뒤 답변이 한 글자씩 실시간으로 타이핑되며 출력됩니다!

### TODO ###
* 코드를 직접 실행해 보시면서 Qwen3 모델의 <think> 태그가 화면에 그대로 노출되어 숨기기
* 나만의 커스텀 도구(DB 조회, 사내 API 등) 연동
