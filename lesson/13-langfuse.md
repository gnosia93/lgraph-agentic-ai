
# Langfuse 워크샵 — Step 1: 첫 모니터링

모델을 붙이고 호출 1건을 보내, 그 호출이 Langfuse에 **trace**로 찍히는 것을 직접 보는 첫걸음입니다.

## 무엇을 보게 되나

`ask()` 함수를 한 번 호출하면 Langfuse 대시보드 **Traces** 탭에 기록이 하나 생깁니다. 트레이스를 열면:

- **Input / Output**: 보낸 질문과 받은 답
- **Latency**: 호출에 걸린 시간(ms)
- **Tokens**: 입력/출력 토큰 수
- **Cost**: 토큰 기반 예상 비용
- **Model**: 사용한 모델 ID

이게 "블랙박스였던 LLM 호출"이 관측 가능해지는 첫 순간입니다.

## 1. Langfuse 준비 (둘 중 하나)

**(A) Langfuse Cloud — 가장 빠름**
1. https://cloud.langfuse.com 가입
2. 프로젝트 생성 → Settings → API Keys 에서 키 발급

**(B) 셀프호스팅 — 로컬 도커**
```bash
# Langfuse를 로컬에 띄우기 (별도 터미널에서)
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker compose up
# http://localhost:3000 접속 → 가입 → 프로젝트/키 생성
```

## 2. 설치

```bash
pip install langfuse boto3
```

## 3. 환경변수 설정

```bash
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_HOST=https://cloud.langfuse.com   # 셀프호스팅이면 http://localhost:3000

# AWS / Bedrock
export AWS_REGION=us-west-2
export BEDROCK_MODEL_ID=us.anthropic.claude-3-5-sonnet-20240620-v1:0
# aws configure 로 자격증명 설정 + 해당 모델 액세스 권한 필요
```

> 모델 ID는 본인 계정에서 액세스 가능한 모델로 교체하세요. (Bedrock 콘솔 → Model access)

## 4. 실행

```bash
python step1_trace_bedrock.py
```

실행 후 Langfuse 대시보드 **Traces** 탭을 새로고침하면 방금 2건이 보입니다.

## 코드의 핵심 3줄

```python
@observe()                                  # ① 함수 호출 1건 = Trace 하나
with langfuse.start_as_current_generation(  # ② LLM 호출 = Generation
        name=..., model=..., input=...) as generation:
    ...
    generation.update(output=..., usage_details=...)  # ③ 출력·토큰 기록 → 비용/지연
```

## 트러블슈팅

- 대시보드에 안 보임 → 스크립트 끝에 `langfuse.flush()` 가 호출됐는지 확인 (전송 버퍼를 비워야 함).
- 인증 오류 → `LANGFUSE_HOST` 가 키를 발급한 인스턴스 주소와 일치하는지 확인.
- Bedrock 권한 오류 → 모델 액세스 승인 + IAM `bedrock:InvokeModel` 권한 확인.

## 다음 단계 (Step 2 미리보기)

- 멀티스텝(에이전트) 트레이싱: `ask()` 안에서 검색→추론→답변을 각각 observation으로 쪼개 보기
- 프롬프트 관리: Langfuse에 프롬프트를 등록하고 버전별로 추적
- 평가(evals): 출력에 점수를 매겨 품질을 시간에 따라 측정

> 참고: Langfuse SDK는 빠르게 업데이트됩니다. API가 다르면 공식 문서(https://langfuse.com/docs)의 Python SDK(v3) 기준을 확인하세요. 본 예제는 v3(OpenTelemetry 기반, `@observe` + `start_as_current_generation`) 기준입니다.
```
