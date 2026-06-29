## Bifrost로 LLM 게이트웨이 구축하기 ##
### Bifrost 란 ###
Bifrost는 Maxim AI가 Go로 만든 고성능 오픈소스 AI 게이트웨이다. OpenAI·Anthropic·AWS Bedrock·Google Vertex·Azure 등 20개 이상의 LLM 프로바이더를 단일 OpenAI 호환 API로 묶어 준다. 애플리케이션 코드에서는 base_url만 Bifrost로 바꾸면, 뒤쪽 프로바이더는 마음껏 교체·조합할 수 있다.

공식 벤치마크 기준 5,000 RPS 부하에서 요청당 약 11µs의 오버헤드만 추가한다고 밝히고 있어, LiteLLM 같은 Python 기반 게이트웨이의 대안으로 자리 잡고 있다. (Content was rephrased for compliance with licensing restrictions — 출처: Bifrost GitHub)

### 핵심 기능 ###
* Unified API: 모든 프로바이더를 OpenAI 호환 스키마로 호출
* Automatic Fallback / Load Balancing: 프로바이더·API 키 단위의 장애 조치와 부하 분산
* Semantic Caching: 의미 기반 응답 캐싱으로 비용·지연 절감
* MCP Gateway: Model Context Protocol 클라이언트/서버 양쪽으로 동작, 외부 도구를 모델에 연결
* Governance: 가상 키, 팀/고객별 예산, 속도 제한, 접근 제어
* Observability: Prometheus 메트릭, 분산 추적, 요청 로그
* Drop-in Replacement: OpenAI/Anthropic/GenAI SDK의 엔드포인트만 교체하면 끝

### 언제 쓰나 ###
* 여러 프로바이더를 쓰는데 SDK가 제각각이라 코드가 지저분해질 때
* 한 모델이 장애 났을 때 다른 모델로 자동 전환하고 싶을 때
* 팀·고객별로 API 키 사용량과 비용을 통제해야 할 때
* Claude Code 같은 도구를 Anthropic 외의 모델(OpenAI, Gemini 등)로 돌리고 싶을 때
* MCP 기반 에이전트를 하나의 엔드포인트로 집약하고 싶을 때

## 설치 및 실행하기 ##
### 1. 설치 및 실행 ###
세 가지 배포 방식이 있다. 가장 빠른 건 NPX다.

#### NPX (30초 안에 기동) ####
```
npx -y @maximhq/bifrost
```
기본 포트 8080에서 게이트웨이와 Web UI가 뜬다.

#### Docker ####
```
docker run -p 8080:8080 \
  -v $(pwd)/data:/app/data \
  maximhq/bifrost
```
data 볼륨에 설정·로그가 영속화된다. 프로덕션에서는 이 방식을 권장한다.

#### Go SDK ####
Go 애플리케이션에 직접 임베드하려면:
```
go get github.com/maximhq/bifrost/core
```
게이트웨이 프로세스 없이 라이브러리로 붙여 쓸 수 있다. 최소 오버헤드가 필요하거나, 사내 서비스에 내장하고 싶을 때 쓴다.

### 2. Web UI로 프로바이더 설정 ###
```
브라우저에서 Web UI를 연다.

open http://localhost:8080
UI에서 할 수 있는 것:

프로바이더 등록: OpenAI / Anthropic / Bedrock / Vertex 등 키 입력
가상 키(Virtual Key) 발급: 실제 API 키를 숨기고, 팀·사용자별로 위임 키를 발급
예산·속도 제한 설정: 월 한도, RPS, 모델별 quota
폴백 체인 구성: "OpenAI → Anthropic → Bedrock" 순으로 재시도
실시간 모니터링: 요청 로그, 지연, 토큰 사용량 대시보드
설정은 내부 저장소에 저장되고, 파일 기반 설정(config.json)이나 API로도 관리할 수 있다.
```

### 3. 첫 번째 호출 ###
OpenAI 호환 엔드포인트가 /v1/chat/completions에 뜬다. 모델 ID는 프로바이더/모델 형식으로 지정한다.
```
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-4o-mini",
    "messages": [{"role": "user", "content": "Hello, Bifrost!"}]
  }'
```

### 4. 드롭인 대체(Drop-in Replacement) ###
기존 SDK 코드를 그대로 두고 base_url만 교체해도 된다.

OpenAI SDK (Python)
```
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/openai",  # 원래: https://api.openai.com
    api_key="bifrost-virtual-key",             # Bifrost에서 발급한 가상 키
)

resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "안녕"}],
)
print(resp.choices[0].message.content)
Anthropic SDK

from anthropic import Anthropic

client = Anthropic(
    base_url="http://localhost:8080/anthropic",  # 원래: https://api.anthropic.com
    api_key="bifrost-virtual-key",
)
```
Google GenAI SDK는 api_endpoint를 http://localhost:8080/genai로 바꾸면 된다.

이 방식의 장점은 코드 한 줄만 바꾸면 관찰성·폴백·캐싱·거버넌스가 전부 자동으로 얹힌다는 점이다.

### 5. 폴백 체인 구성 (예시) ###
Web UI 또는 설정 파일에서 다음과 같이 선언한다.
```
{
  "providers": [
    { "name": "openai", "api_keys": ["sk-..."] },
    { "name": "anthropic", "api_keys": ["sk-ant-..."] },
    { "name": "bedrock", "region": "ap-northeast-2" }
  ],
  "fallbacks": {
    "gpt-4o-mini": [
      "openai/gpt-4o-mini",
      "anthropic/claude-3-5-haiku",
      "bedrock/anthropic.claude-3-haiku-20240307-v1:0"
    ]
  }
}
```
OpenAI가 5xx·타임아웃·레이트 리밋을 내면 Anthropic → Bedrock 순으로 자동 재시도한다. 클라이언트 입장에서는 아무 변화가 없다.

### 6. MCP 게이트웨이로 활용 ###
Bifrost는 MCP 클라이언트와 서버 양쪽 역할을 한다. 즉:

* 위쪽(클라이언트 쪽): Claude Code, Claude Desktop 같은 MCP 클라이언트가 Bifrost 하나만 바라보게 한다.
* 아래쪽(서버 쪽): 파일시스템, 웹 검색, DB, 사내 API 같은 MCP 서버 여러 개를 Bifrost가 물고 있는다.
* 클라이언트는 /mcp 엔드포인트 하나로 전체 도구 카탈로그에 접근하게 된다. 도구가 수백 개로 늘어나면 토큰 비용이 폭증하는데, Bifrost의 Code Mode는 도구 호출을 코드 형태로 압축해 토큰 소비를 크게 줄여 준다. (Content was rephrased for compliance with licensing restrictions — 출처: Maxim AI 블로그)

### 7. 관찰성(Observability) ###
* Prometheus 엔드포인트(/metrics)로 요청량·지연·에러율·토큰 사용량을 수집
* 분산 추적(OpenTelemetry) 연동
* Web UI의 Logs 탭에서 요청별 입출력·프로바이더·폴백 경로 확인
* Grafana 대시보드에 그대로 꽂아 모델별 비용·지연을 모니터링할 수 있다.

### 운영하면서 주의할 점 ###
* 가상 키 관리: 실제 프로바이더 키는 Bifrost 내부에만 두고, 애플리케이션에는 가상 키만 배포하는 패턴이 안전하다. 유출 시 해당 가상 키만 회수하면 된다.
* 단일 장애점(SPOF): 게이트웨이 자체가 죽으면 전체 트래픽이 영향을 받는다. 프로덕션에서는 여러 인스턴스를 띄우고 앞단에 LB를 두거나, Enterprise의 클러스터 모드를 쓴다.
* 시맨틱 캐시 주의: 의미 기반 캐싱은 비용을 크게 줄여 주지만, 시점성 정보(오늘 환율, 최신 뉴스 등)에는 부적합하다. 캐시 제외 규칙을 명시적으로 설정하자.
* Bedrock 등 리전 제약: 모델 ID 형식이 프로바이더마다 다르다. Bedrock은 리전·크로스 리전 프리픽스까지 정확히 맞춰야 한다.

### 마무리 ###
Bifrost의 매력은 "프로바이더 추상화 + 게이트웨이 기능"을 코드 변경 없이 얹을 수 있다는 점이다. 앞서 다룬 promptfoo로 모델을 평가해 최적 조합을 찾았다면, Bifrost로 그 조합을 프로덕션 라우팅 레이어로 내릴 수 있다. 평가(promptfoo) → 배포(Bifrost) 흐름으로 이어가면 LLM 스택을 소프트웨어처럼 다루는 파이프라인이 완성된다.

## 레퍼런스 ##
* https://github.com/maximhq/bifrost
* https://www.getmaxim.ai/bifrost/blog

## todo ##
* 클러스터 배포,
* MCP 연결 실습,
* Langchain 통합
  
