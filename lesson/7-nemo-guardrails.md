
NVIDIA NeMo Guardrails는 LLM 기반 애플리케이션의 입력(Prompt Injection, 욕설 등)과 출력(환각, 탈옥, 오프토픽 등)을 제어하기 위한 프로그래머블 가드레일 오픈소스 툴킷입니다. 엔지니어링 관점에서 결정적 제어(Deterministic Control) 파이프라인을 구축할 때 아주 유용합니다.
NeMo Guardrails의 핵심 아키텍처와 구체적인 사용법을 3단계로 핵심만 정리해 드릴게요.

### 📂 1. 프로젝트 기본 구조 세팅 ###
NeMo Guardrails는 Python 코드 내부가 아니라, **별도의 설정 폴더(config)**를 만들고 그 안에 **YAML 파일(모델 및 레일 설정)**과 **Colang 파일(대화 흐름 정의)**을 선언하는 방식으로 동작합니다.
프로젝트 디렉토리를 다음과 같이 구성합니다.

```
my_guardrails_app/
├── config/
│   ├── config.yml      # LLM 설정 및 가드레일 활성화 옵션
│   └── rails.co        # Colang 2.0 기반의 가드레일 흐름(Flow) 정의
└── main.py             # 실행할 Python 메인 스크립트
```

### 🛠️ 2. 핵심 설정 파일 작성 ###
#### ① config/config.yml (모델 및 전역 설정) ####
어떤 LLM 엔진을 사용할지, 어떤 레일(인풋, 아웃풋 등)을 켤지 선언합니다.
```
models:
  - type: main
    engine: openai
    model: gpt-4o

# 활성화할 가드레일 유형 정의
rails:
  config:
    self_check_input: true    # 사용자 입력 검사 (Jailbreak 등)
    self_check_output: true   # 모델 출력 검사 (환각, 욕설 등)
```

#### ② config/rails.co (Colang 대화 흐름 제어) ###
NVIDIA가 개발한 Colang이라는 인터프리터 언어를 사용하여 "특정 주제나 위험 발언이 들어오면 LLM 본체로 보내지 말고 차단하라"는 룰셋을 정의합니다.

```
# 1. 사용자의 의도(Intent) 정의 (예시 문장 학습)
define user ask politics
  "정치에 대해 어떻게 생각해?"
  "대통령 선거 누가 이길 것 같아?"
  "특정 정당 지지해?"

# 2. 봇의 대응 정의
define bot refuse to talk politics
  "죄송합니다, 저는 정치적인 주제에 대해서는 답변할 수 없도록 설정되어 있습니다."

# 3. 결정적 제어 흐름(Flow) 매핑
define flow politics restriction
  user ask politics
  bot refuse to talk politics
```

### 3. Python 애플리케이션 연동 (main.py) ###
설정 폴더를 RailsConfig로 로드한 뒤, LLMRails 인스턴스를 통해 LLM을 호출합니다. 겉보기에는 OpenAI API나 LangChain을 쓰는 것과 유사하지만, 내부적으로 가드레일 파이프라인을 먼저 거치게 됩니다.

```
import os
from nemoguardrails import RailsConfig, LLMRails

# OpenAI API Key 설정 (혹은 vLLM 등 로컬 엔드포인트 바인딩 가능)
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

def main():
    # 1. 설정 폴더 로드
    config = RailsConfig.from_path("./config")
    
    # 2. 가드레일 앱 생성
    app = LLMRails(config)

    # 3. 테스트 1: 정상적인 질문
    response = app.generate(messages=[
        {"role": "user", "content": "양자 컴퓨터의 이온 트랩 방식이 뭐야?"}
    ])
    print("일반 답변:", response["content"])

    print("-" * 30)

    # 4. 테스트 2: 차단된 정치 관련 질문 (rails.co에서 정의한 흐름 발동)
    blocked_response = app.generate(messages=[
        {"role": "user", "content": "다음 대선 후보 중에 누가 제일 유리해?"}
    ])
    print("가드레일 작동 답변:", blocked_response["content"])

if __name__ == "__main__":
    main()
```

### 실무 엔지니어링 팁 (Advanced) ###
*	자체 임베딩/NIM 연동: config.yml에 엔비디아의 NeMo Retriever나 자체 임베딩 모델을 세팅하여 사용자의 의도(Intent)를 더 정확하게 분류(Classification)할 수 있습니다.
*	Custom Python Actions: Colang 내부에서 특정 조건이 발동했을 때 외부 DB를 조회하거나, 비속어 필터링 API를 찌르는 등 **Python 함수(Action)**를 데코레이터(@action) 형태로 등록해 런타임 제어 흐름에 끼워 넣을 수 있습니다.


### Colang flow ###
* RAG 파이프라인의 환각 차단,
* Prompt Injection 방어
* 특정 도메인 탈탈출 방지 



