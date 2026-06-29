## Langfuse 평가(Evals): 출력 품질을 점수로 추적하기 ##

### 1. 왜 평가가 필요한가 ###

호출을 기록하고 프롬프트를 버전으로 관리하는 방법을 다뤘다. 그러나 기록만으로는 품질이 좋아졌는지 판단할 수 없다. LLM 출력은 정답이 하나로 정해지지 않는 경우가 많고, 같은 질문에도 표현이 매번 달라지므로, 단순 일치 비교로는 품질을 측정하기 어렵다. 프롬프트를 바꾸거나 모델을 교체했을 때 그 변경이 개선인지 후퇴인지 확인하려면, 출력에 정량적인 점수를 부여하고 그 점수를 시간과 버전에 따라 추적해야 한다.

Langfuse에서 이 점수를 Score라고 한다. Score는 Trace나 Observation에 부여되는 수치 또는 범주값이며, 품질, 정확성, 사용자 만족도 등 무엇이든 측정 대상이 될 수 있다. Score를 부여하는 방법은 세 가지다. 사람이 직접 매기는 방식, 프로그램이나 LLM이 자동으로 매기는 방식, 그리고 데이터셋을 기준으로 일괄 평가하는 방식이다.

### 2. Score의 종류 ###

Score에는 데이터 타입이 있다. 수치형(NUMERIC)은 0부터 1까지의 점수처럼 연속값을 기록한다. 범주형(CATEGORICAL)은 "정확함", "부정확함" 같은 라벨을 기록한다. 불리언(BOOLEAN)은 통과 여부를 기록한다. 측정하려는 대상에 맞는 타입을 선택한다.

각 Score에는 이름이 있다. 같은 이름의 Score를 여러 Trace에 부여하면, 대시보드에서 그 이름의 점수가 시간에 따라 어떻게 변하는지 집계해 볼 수 있다. 예를 들어 relevance라는 이름으로 모든 응답의 관련성 점수를 기록하면, 프롬프트 버전 배포 전후로 relevance 평균이 어떻게 달라졌는지 비교할 수 있다.

### 3. 사람이 직접 평가하기 ###

가장 단순한 방법은 대시보드에서 사람이 직접 점수를 매기는 것이다. Trace를 열어 출력을 확인하고, 품질을 점수나 라벨로 기록한다. 소수의 응답을 정밀하게 검토하거나, 자동 평가의 기준을 세우기 위한 정답 데이터를 만들 때 사용한다. 별도 코드가 필요 없으며, 운영 중 수집된 실제 트래픽을 표본으로 검토하는 데 적합하다.

### 4. LLM으로 자동 평가하기 (LLM-as-judge) ###

응답이 많아지면 사람이 모두 검토할 수 없다. 이때 다른 LLM에게 출력을 평가하게 한다. 평가용 모델에 원래 질문과 생성된 답변을 주고, 정해진 기준에 따라 점수를 매기게 한 뒤 그 점수를 Score로 기록한다.

```python
import os
import json
import boto3
from langfuse import observe, get_client

langfuse = get_client()
REGION = os.environ.get("AWS_REGION", "us-west-2")
MODEL_ID = os.environ.get(
    "BEDROCK_MODEL_ID", "us.anthropic.claude-3-5-sonnet-20240620-v1:0"
)
bedrock = boto3.client("bedrock-runtime", region_name=REGION)


def judge(question: str, answer: str) -> dict:
    """평가용 모델에 답변의 관련성을 0~1로 채점하게 한다."""
    prompt = (
        "다음 질문과 답변을 평가하라. 답변이 질문에 얼마나 관련되고 정확한지 "
        "0.0에서 1.0 사이 점수로 매기고, 한 문장 근거를 제시하라. "
        'JSON 형식으로만 출력하라: {"score": 숫자, "reason": "근거"}\n\n'
        f"질문: {question}\n답변: {answer}"
    )
    response = bedrock.converse(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 256, "temperature": 0.0},
    )
    text = response["output"]["message"]["content"][0]["text"]
    return json.loads(text)


@observe()
def answer_and_evaluate(question: str, answer: str):
    # 답변에 대해 LLM 평가를 수행
    result = judge(question, answer)

    # 평가 점수를 현재 Trace에 Score로 부여
    langfuse.score_current_trace(
        name="relevance",
        value=result["score"],     # 0.0 ~ 1.0
        comment=result["reason"],
    )
    return result


if __name__ == "__main__":
    q = "환불 규정이 어떻게 되나요?"
    a = "환불은 구매 후 14일 이내에 영수증과 함께 가능합니다."
    print(answer_and_evaluate(q, a))
    langfuse.flush()
```

평가 모델에는 온도를 0으로 두어 채점의 일관성을 높인다. score_current_trace는 현재 활성화된 Trace에 Score를 부여하므로, 평가 결과가 해당 응답의 트레이스에 함께 기록된다. 평가 기준을 JSON으로 받으면 점수와 근거를 구조적으로 추출할 수 있다.

LLM 평가는 자동화할 수 있다는 장점이 있으나, 평가 모델 자체가 완벽하지 않으므로 사람 평가로 그 신뢰도를 주기적으로 검증하는 것이 좋다.

### 5. 데이터셋으로 일괄 평가하기 ###

운영 트래픽에 대한 평가와 별개로, 고정된 테스트 입력 묶음에 대해 반복 평가가 필요하다. 프롬프트나 모델을 바꿀 때마다 같은 입력에 실행해 결과를 비교하기 위해서다. Langfuse에서는 이를 데이터셋과 실험(run)으로 관리한다.

데이터셋은 입력과 기대 출력의 쌍으로 구성한다. 기대 출력은 정답 비교나 채점의 기준으로 사용한다.

```python
from langfuse import get_client

langfuse = get_client()

# 데이터셋 생성
langfuse.create_dataset(name="qa-eval")

# 항목 추가 (입력과 기대 출력)
langfuse.create_dataset_item(
    dataset_name="qa-eval",
    input="환불 규정이 어떻게 되나요?",
    expected_output="구매 후 14일 이내, 영수증 필요",
)
```

이후 프롬프트나 모델을 바꿔 데이터셋 전체를 실행하고 각 항목에 점수를 부여하면, 실험(run) 단위로 결과가 묶여 기록된다. 대시보드에서 run 간 점수를 비교해 어떤 설정이 더 나은 결과를 내는지 확인할 수 있다. 데이터셋 평가는 코드 변경의 효과를 배포 전에 측정하는 회귀 테스트 역할을 한다.

## 6. 평가를 운영에 연결하기

세 가지 평가 방식은 함께 쓰인다. 운영 트래픽 일부에 LLM 평가를 적용해 품질을 상시 모니터링하고, 그중 점수가 낮은 응답을 사람이 검토하며, 검토에서 확인된 사례를 데이터셋에 추가해 다음 변경의 회귀 테스트로 활용한다. 3편의 프롬프트 버전과 연결하면, 특정 프롬프트 버전 배포 이후 평가 점수가 어떻게 변했는지 추적할 수 있고, 점수가 하락하면 이전 버전으로 롤백하는 판단 근거가 된다.

이로써 관측(트레이싱), 관리(프롬프트), 측정(평가)이 하나의 순환을 이룬다. 호출을 기록하고, 품질을 측정하고, 그 결과로 프롬프트와 모델을 개선하며, 개선의 효과를 다시 측정한다.

