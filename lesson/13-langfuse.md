
### 1. Langfuse란? 왜 필요한가 ###

Langfuse는 LLM 애플리케이션을 위한 오픈소스 옵저버빌리티 플랫폼이다. LLM 호출과 그 호출을 둘러싼 처리 과정을 기록하고, 입력·출력·지연·토큰·비용을 호출 단위로 조회할 수 있게 한다.

기존의 로그나 APM(CloudWatch, Datadog 등)은 상태 코드, 응답 시간, 에러율 같은 지표를 다룬다. 동작이 결정적이고 처리 경로가 고정된 일반 백엔드에는 이것으로 충분하다. LLM 애플리케이션은 다음 성질 때문에 동일한 방식으로 관측하기 어렵다.

LLM 호출은 비결정적이다. 같은 입력에 대해 출력이 매번 달라지므로, 문제가 된 응답을 사후에 재현할 수 없다. 그 시점의 입력과 출력을 기록해두지 않으면 분석할 근거가 남지 않는다. 또한 동작 품질이 코드가 아니라 프롬프트와 모델에 의해 결정된다. 코드 변경 없이 프롬프트 문구나 모델 버전을 바꾸는 것만으로 결과가 달라지므로, 출력 품질 자체를 추적 대상으로 삼아야 한다. 비용은 호출마다 토큰량에 비례해 발생하며, 기능·사용자·프롬프트 단위로 분해하지 않으면 어디서 비용이 발생하는지 파악할 수 없다. 마지막으로 RAG나 에이전트 구조에서는 하나의 요청이 검색, 컨텍스트 구성, 추론, 도구 호출, 생성 등 여러 단계로 나뉜다. 최종 출력이 잘못되었을 때 어느 단계에서 문제가 발생했는지 확인하려면 단계 간 흐름을 하나로 연결해 기록해야 한다.

Langfuse는 이러한 요구를 충족하기 위해 다음 기능을 제공한다. 트레이싱은 요청 한 건의 전체 처리 흐름을 입력, 출력, 지연, 토큰, 비용과 함께 기록한다. 프롬프트 관리는 프롬프트를 코드와 분리해 버전 단위로 관리하고 트레이스와 연결한다. 평가(Evals)는 출력 품질에 점수를 부여해 시간에 따른 변화를 추적한다. 대시보드는 비용, 지연, 사용량, 품질 지표를 집계해 보여준다. 오픈소스이므로 데이터를 외부로 전송하기 어려운 환경에서는 셀프호스팅으로 운영할 수 있다.

Langfuse의 데이터 모델은 네 가지 개념으로 구성된다. Trace는 요청 한 건의 전체 기록이다. Observation은 Trace 내부의 개별 처리 단계를 나타낸다. Generation은 LLM 호출을 나타내는 Observation의 한 종류이며, 모델 정보와 토큰·비용이 함께 기록된다. Session은 여러 Trace를 하나의 대화 단위로 묶는다. 사용자의 한 요청이 하나의 Trace가 되고, 그 안의 처리 단계가 Observation으로 기록되며, 그중 모델을 호출한 단계가 Generation이 된다. 여러 요청이 이어지는 대화는 Session으로 묶인다.

이 문서에서는 가장 기본 단위인 Trace 하나와 Generation 하나를 생성해, 단일 모델 호출이 어떻게 조회 가능한 데이터로 기록되는지 확인한다.


### 2. 설치 ###

#### 2.1 Langfuse 인스턴스 준비 ###

Langfuse를 사용하려면 데이터를 수집할 서버와 인증 키가 필요하다. Langfuse Cloud를 사용하는 경우, cloud.langfuse.com에 가입하고 프로젝트를 생성한 뒤 Settings의 API Keys 메뉴에서 Public Key와 Secret Key를 발급한다. 별도 설치 없이 가장 빠르게 시작할 수 있다.
셀프호스팅으로 운영하는 경우, 아래와 Docker Compose로 쉽게 설치할 수 있다.

```bash
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker compose up
```
기동 후 http://localhost:3000 에 접속해 계정을 만들고 프로젝트와 API 키를 생성한다.

#### 2.2 SDK 설치 및 환경 변수 설정 ####

이 문서는 Amazon Bedrock을 기준으로 한다. Bedrock 호출에는 boto3를, 트레이싱에는 langfuse를 설치한다.

```bash
pip install langfuse boto3
```

Langfuse SDK는 다음 환경 변수를 자동으로 읽어 초기화된다. Bedrock 호출에 필요한 AWS 설정도 함께 지정한다.

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_HOST=https://cloud.langfuse.com   # 셀프호스팅이면 http://localhost:3000

export AWS_REGION=us-west-2
export BEDROCK_MODEL_ID=us.anthropic.claude-3-5-sonnet-20240620-v1:0
```

BEDROCK_MODEL_ID는 사용 중인 계정에서 접근 가능한 모델로 지정한다. 모델 접근 권한은 Bedrock 콘솔의 Model access에서 승인한다.


### 3. 트레이스 설정 ###

LLM 호출 코드에 Langfuse를 연결한다. 함수 단위 트레이싱에는 observe 데코레이터를, LLM 호출 기록에는 start_as_current_generation 컨텍스트 매니저를 사용한다.

[step1_trace_bedrock.py]
```python
import os
import boto3
from langfuse import observe, get_client

langfuse = get_client()

REGION = os.environ.get("AWS_REGION", "us-west-2")
MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID",
    "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
)
bedrock = boto3.client("bedrock-runtime", region_name=REGION)

@observe()
def ask(question: str) -> str:
    with langfuse.start_as_current_generation(
        name="bedrock-converse",
        model=MODEL_ID,
        input=question,
    ) as generation:

        response = bedrock.converse(
            modelId=MODEL_ID,
            messages=[{"role": "user", "content": [{"text": question}]}],
            inferenceConfig={"maxTokens": 512, "temperature": 0.7},
        )

        answer = response["output"]["message"]["content"][0]["text"]
        usage = response.get("usage", {})

        generation.update(
            output=answer,
            usage_details={
                "input": usage.get("inputTokens", 0),
                "output": usage.get("outputTokens", 0),
                "total": usage.get("totalTokens", 0),
            },
        )

    return answer


if __name__ == "__main__":
    print(ask("옵저버빌리티가 필요한 이유를 한 문장으로 설명해줘."))
    langfuse.flush()
```

코드의 동작은 다음과 같다. observe 데코레이터는 ask 함수의 호출 한 건을 하나의 Trace로 만든다. start_as_current_generation은 해당 Trace 안에 LLM 호출을 나타내는 Generation을 생성하며, 호출 전에 모델 ID와 입력을 기록한다. Bedrock 응답을 받은 뒤 generation.update로 출력과 토큰 사용량을 채우면, Langfuse가 이를 기반으로 비용과 지연을 계산한다.

마지막의 langfuse.flush 호출은 필수다. Langfuse는 성능을 위해 데이터를 버퍼에 모아 비동기로 전송하므로, 짧게 종료되는 스크립트에서는 flush로 버퍼를 비우지 않으면 데이터가 전송되지 않는다.

  
### 4. 관측하기 ###

아래 스크립트를 실행한다.

```bash
python step1_trace_bedrock.py
```

실행 후 Langfuse 대시보드의 Traces 탭을 열면 방금 실행한 호출이 하나의 Trace로 기록되어 있다. Trace를 선택하면 다음 정보를 확인할 수 있다. 입력과 출력 전문, 호출에 걸린 지연 시간, 입력·출력 토큰 수, 토큰 기반 예상 비용, 사용한 모델 ID, 그리고 Trace 내부에서 Generation이 차지한 구간을 보여주는 타임라인이다. 
이 시점에서 단일 모델 호출이 조회 가능한 데이터로 전환된다. 어떤 입력에 대해 어떤 출력이, 어떤 모델로, 얼마의 비용과 시간으로 발생했는지를 호출 단위로 확인할 수 있다.

요청에 user_id나 session_id를 부여하면 사용자별 비용 집계나 대화 세션 단위 조회가 가능하다.

### 5. Advanced 트레이싱 ###

* [멀티스텝 트레이싱](https://github.com/gnosia93/langgraph-agentic-ai/blob/main/lesson/13-langfuse-multi.md)
* [프롬프트 관리](https://github.com/gnosia93/langgraph-agentic-ai/blob/main/lesson/13-langfuse-prompt.md)

## 레퍼런스 ##
* https://langfuse.com/docs
