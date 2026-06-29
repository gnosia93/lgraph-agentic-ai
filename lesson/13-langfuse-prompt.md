## Langfuse 프롬프트 관리: 코드에서 분리하고 버전으로 추적하기 ##


### 1. 프롬프트를 코드에 두면 생기는 문제 ###

지금까지의 예제에서 프롬프트는 코드 안에 문자열로 하드코딩되어 있었다. 동작에는 문제가 없지만, 운영 단계에서 다음 문제가 발생한다.

프롬프트를 수정하려면 코드를 변경하고 다시 배포해야 한다. 프롬프트 문구 하나를 바꾸는 작업이 코드 배포 절차를 거쳐야 하므로, 비개발 직군은 직접 수정할 수 없고 반복 실험의 속도도 느리다. 또한 어떤 시점에 어떤 프롬프트가 사용되었는지 코드 이력을 일일이 추적해야 하며, 특정 응답이 어느 프롬프트 버전에서 나왔는지 트레이스와 연결되지 않는다. 프롬프트를 바꿨을 때 품질이 좋아졌는지 나빠졌는지 비교하려면 버전 간 결과를 묶어 볼 수단이 필요하다.

Langfuse의 프롬프트 관리는 프롬프트를 코드에서 분리해 Langfuse에 저장하고, 버전과 레이블로 관리하며, 각 프롬프트가 사용된 트레이스와 연결한다.

### 2. 동작 방식 ###

프롬프트는 이름으로 식별되며, 내용이 바뀔 때마다 버전이 증가한다. 각 버전에는 레이블을 붙일 수 있다. 예를 들어 production 레이블이 붙은 버전을 운영에서 사용하고, 새 버전을 만들어 테스트한 뒤 검증이 끝나면 production 레이블을 옮기는 방식으로 배포를 제어한다. 애플리케이션은 코드 변경 없이 특정 레이블의 프롬프트를 가져와 사용한다.

프롬프트에는 변수를 포함할 수 있다. 변수는 이중 중괄호로 표기하며, 실행 시점에 실제 값으로 치환한다. SDK는 가져온 프롬프트를 캐싱하므로 매 호출마다 네트워크 요청이 발생하지 않는다.

### 3. 프롬프트 생성 ###

프롬프트는 Langfuse UI에서 만들거나 SDK로 생성할 수 있다. 다음은 SDK로 텍스트 프롬프트를 생성하는 예이다. context와 question을 변수로 둔다.

```python
from langfuse import get_client

langfuse = get_client()

langfuse.create_prompt(
    name="qa-prompt",
    prompt=(
        "다음 문서를 근거로 질문에 답하라.\n\n"
        "문서:\n{{context}}\n\n"
        "질문: {{question}}"
    ),
    labels=["production"],   # 이 버전을 production으로 지정
    config={"model": "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
            "temperature": 0.3},
)
```

config에는 모델 ID나 온도 같은 호출 파라미터를 함께 저장할 수 있다. 프롬프트와 그에 맞는 파라미터를 한곳에서 관리하기 위한 용도이다. 같은 이름으로 다시 생성하면 새 버전이 만들어진다.

### 4. 프롬프트 사용과 트레이스 연결 ###

애플리케이션에서는 이름과 레이블로 프롬프트를 가져온 뒤, 변수를 채워 모델에 전달한다. 이때 가져온 프롬프트 객체를 Generation에 연결하면, 대시보드에서 해당 응답이 어떤 프롬프트 버전으로 생성되었는지 확인할 수 있다.

```python
import os
import boto3
from langfuse import observe, get_client

langfuse = get_client()
bedrock = boto3.client("bedrock-runtime",
                       region_name=os.environ.get("AWS_REGION", "us-west-2"))


@observe()
def answer_question(question: str, context: str) -> str:
    # 1. production 레이블의 프롬프트를 가져온다 (캐싱됨)
    prompt = langfuse.get_prompt("qa-prompt", label="production")

    # 2. 변수를 채워 최종 프롬프트 문자열을 만든다
    compiled = prompt.compile(context=context, question=question)

    # 3. 프롬프트에 저장된 config에서 모델 파라미터를 읽는다
    model_id = prompt.config["model"]
    temperature = prompt.config.get("temperature", 0.3)

    # 4. Generation에 프롬프트를 연결해 호출을 기록한다
    with langfuse.start_as_current_generation(
        name="generate",
        model=model_id,
        input=compiled,
        prompt=prompt,   # 이 응답이 어떤 프롬프트 버전인지 연결
    ) as generation:
        response = bedrock.converse(
            modelId=model_id,
            messages=[{"role": "user", "content": [{"text": compiled}]}],
            inferenceConfig={"maxTokens": 512, "temperature": temperature},
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
    ctx = "환불은 구매 후 14일 이내에 가능하며, 영수증이 필요하다."
    print(answer_question("환불 규정이 어떻게 되나요?", ctx))
    langfuse.flush()
```

get_prompt는 production 레이블이 가리키는 현재 버전을 가져온다. 프롬프트 내용을 수정해 새 버전을 만들고 production 레이블을 그 버전으로 옮기면, 이 코드는 수정 없이 새 프롬프트를 사용한다. 배포 절차를 거치지 않고 프롬프트만 교체할 수 있다. start_as_current_generation에 prompt를 전달했으므로, 대시보드의 해당 Generation에서 사용된 프롬프트 버전이 함께 표시된다.

## 5. 버전 비교와 품질 추적

프롬프트를 버전으로 관리하면 변경의 효과를 측정할 수 있다. 새 버전을 만들어 일부 트래픽에 적용하거나 테스트 데이터셋에 실행한 뒤, 버전별로 응답을 비교한다. 응답 품질에 점수를 부여하면(다음 편의 평가 기능) 어떤 버전이 더 나은 결과를 내는지 수치로 확인할 수 있다. 프롬프트 버전과 트레이스가 연결되어 있으므로, 특정 버전 배포 이후 품질이나 비용이 어떻게 변했는지도 추적할 수 있다.

이 방식은 프롬프트 변경을 코드 변경과 분리해, 프롬프트를 독립적으로 실험하고 배포하고 되돌릴 수 있게 한다. 운영 중 품질 저하가 확인되면 production 레이블을 이전 버전으로 되돌리는 것으로 즉시 롤백할 수 있다.


