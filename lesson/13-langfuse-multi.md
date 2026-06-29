# Langfuse 멀티스텝 트레이싱: 다단계 호출의 내부 들여다보기

## 1. 단일 호출 트레이싱의 한계

1편에서는 LLM 호출 한 건을 하나의 Trace로 기록했다. 입력, 출력, 지연, 토큰, 비용을 호출 단위로 확인할 수 있었다. 그러나 실제 애플리케이션은 단일 호출로 끝나지 않는다.

RAG는 사용자 질문으로 문서를 검색하고, 검색 결과를 컨텍스트로 구성한 뒤, 그 컨텍스트를 포함해 모델을 호출한다. 에이전트는 추론으로 다음 행동을 정하고, 도구를 호출하고, 그 결과를 관찰한 뒤 다시 추론한다. 이런 구조에서 최종 출력이 잘못되었을 때, 호출 전체를 하나의 덩어리로만 기록하면 어느 단계에서 문제가 발생했는지 알 수 없다. 검색이 엉뚱한 문서를 가져온 것인지, 검색은 정확했으나 모델이 그 컨텍스트를 잘못 사용한 것인지 구분되지 않는다.

멀티스텝 트레이싱은 하나의 요청을 구성하는 각 단계를 개별 Observation으로 분리해 기록한다. 단계별 입력과 출력, 지연을 따로 확인할 수 있으므로 문제 지점을 특정할 수 있다.

## 2. Trace와 Observation의 계층 구조

Langfuse의 Observation은 중첩될 수 있다. 최상위 함수가 하나의 Trace가 되고, 그 안에서 호출되는 하위 단계들이 자식 Observation으로 기록된다. Observation에는 두 종류를 사용한다. LLM 호출이 아닌 일반 처리 단계(검색, 컨텍스트 구성 등)는 Span으로 기록하고, LLM 호출은 Generation으로 기록한다. Generation에만 모델 정보와 토큰·비용이 붙는다.

이 계층은 코드의 호출 구조를 그대로 따른다. observe 데코레이터가 적용된 함수 안에서 start_as_current_span이나 start_as_current_generation을 호출하면, 생성된 Observation은 현재 활성화된 Trace의 자식으로 자동 연결된다. 별도로 부모를 지정할 필요가 없다.

## 3. 예제: 검색-추론-답변 파이프라인

문서를 검색하고, 검색 결과를 바탕으로 답변을 생성하는 간단한 RAG 형태의 파이프라인을 트레이싱한다. 검색 단계는 Span으로, 모델 호출 단계는 Generation으로 기록한다.

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

# 검색 대상 문서 (실제로는 벡터 DB나 검색 엔진을 사용한다)
DOCUMENTS = {
    "환불": "환불은 구매 후 14일 이내에 가능하며, 영수증이 필요하다.",
    "배송": "배송은 영업일 기준 2~3일 소요되며, 도서산간은 추가 1일이 걸린다.",
    "교환": "교환은 제품 하자 시 무상으로 처리되며, 단순 변심은 배송비가 부과된다.",
}


def retrieve(question: str) -> str:
    """질문에 해당하는 문서를 검색한다. 검색 단계를 Span으로 기록한다."""
    with langfuse.start_as_current_span(
        name="retrieve",
        input=question,
    ) as span:
        matched = [text for key, text in DOCUMENTS.items() if key in question]
        context = "\n".join(matched) if matched else "관련 문서를 찾지 못했다."
        span.update(output=context)
        return context


def generate(question: str, context: str) -> str:
    """검색된 컨텍스트를 포함해 모델을 호출한다. LLM 호출을 Generation으로 기록한다."""
    prompt = f"다음 문서를 근거로 질문에 답하라.\n\n문서:\n{context}\n\n질문: {question}"

    with langfuse.start_as_current_generation(
        name="generate",
        model=MODEL_ID,
        input=prompt,
    ) as generation:
        response = bedrock.converse(
            modelId=MODEL_ID,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": 512, "temperature": 0.3},
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


@observe()
def answer_question(question: str) -> str:
    """검색과 생성을 순서대로 수행한다. 이 함수 호출 한 건이 하나의 Trace가 된다."""
    context = retrieve(question)
    answer = generate(question, context)
    return answer


if __name__ == "__main__":
    print(answer_question("환불 규정이 어떻게 되나요?"))
    langfuse.flush()
```

answer_question 함수에 observe 데코레이터가 적용되어 있으므로 이 함수의 호출 한 건이 하나의 Trace가 된다. 그 안에서 호출되는 retrieve는 Span으로, generate는 Generation으로 기록되며, 둘 다 같은 Trace의 자식 Observation으로 자동 연결된다. retrieve가 LLM을 호출하지 않으므로 Span으로 기록했고, generate는 모델을 호출하므로 토큰과 비용을 기록할 수 있는 Generation으로 기록했다.

## 4. 대시보드에서 확인하기

스크립트를 실행한 뒤 Langfuse 대시보드에서 해당 Trace를 열면, answer_question 아래에 retrieve와 generate가 트리 구조로 표시된다. 각 단계를 선택하면 그 단계의 입력과 출력, 지연을 개별적으로 확인할 수 있다.

이 구조에서 문제를 진단하는 방식은 다음과 같다. 답변이 부정확할 때, retrieve의 출력을 먼저 확인한다. 검색된 컨텍스트가 질문과 무관하다면 문제는 검색 단계에 있다. 검색 결과는 정확한데 답변이 틀렸다면 문제는 생성 단계, 즉 프롬프트나 모델에 있다. 단일 호출로만 기록했다면 이 두 경우를 구분할 수 없었겠지만, 단계를 분리하면 원인을 한정할 수 있다.

각 단계의 지연도 따로 측정되므로 어느 단계가 전체 응답 시간을 지배하는지 파악할 수 있다. 검색이 느린지, 모델 호출이 느린지에 따라 최적화 대상이 달라진다.

## 5. 단계가 더 늘어나는 경우

에이전트처럼 단계 수가 가변적이거나 중첩이 깊어지는 경우에도 동일한 방식이 적용된다. 함수가 다른 함수를 호출하고 그 안에서 다시 Span이나 Generation을 생성하면, Observation의 중첩 구조도 그에 맞춰 깊어진다. 도구 호출을 Span으로, 각 추론 단계의 LLM 호출을 Generation으로 기록하면 에이전트가 어떤 순서로 무엇을 판단하고 실행했는지 전체 흐름이 하나의 Trace에 남는다.

여러 단계에서 같은 사용자나 같은 대화에 속한 요청을 묶으려면 Trace에 user_id와 session_id를 부여한다. 이렇게 하면 대시보드에서 사용자별 비용이나 세션 단위의 전체 대화 흐름을 조회할 수 있다.

## 6. 다음 편

이 편에서는 다단계 호출을 Span과 Generation으로 분리해 기록하고, 단계별로 문제를 진단하는 방법을 다뤘다. 다음 편에서는 프롬프트를 코드에서 분리해 Langfuse에서 버전 단위로 관리하고, 각 버전이 어떤 트레이스와 연결되는지 추적하는 프롬프트 관리를 다룬다.
