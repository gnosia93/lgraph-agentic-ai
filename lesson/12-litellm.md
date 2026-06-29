
## AI 에이전트 개발, 모델 파편화의 늪에서 탈출하기: LiteLLM 활용기 ##

AI 에이전트나 자동화 시스템을 개발하다 보면 반드시 마주하는 벽이 있습니다. 바로 **'모델 파편화'**입니다. 프로젝트 초기에 OpenAI의 GPT-4o로 설계를 시작했지만, 비용 최적화를 위해 Claude나 로컬 모델로 전환하려고 할 때마다 코드를 전부 뜯어고쳐야 하는 상황, 한 번쯤 겪어보셨을 겁니다.
이 지루하고 반복적인 노가다를 단번에 해결해 주는 도구가 바로 LiteLLM입니다.

### 1. LiteLLM 이란 ? ###
LiteLLM은 한마디로 **"모든 LLM API를 하나로 묶어주는 표준 어댑터"**입니다.
OpenAI, Anthropic, Google Gemini, HuggingFace 등 모델마다 각기 다른 API 문법을 가지고 있습니다. LiteLLM은 이 복잡한 인터페이스를 OpenAI의 표준 포맷으로 통일해 줍니다. 즉, 어떤 모델을 쓰든 라이브러리 교체 없이 코드 한 줄(모델 이름)만 바꾸면 끝납니다.

### 2. 왜 실무자들은 LiteLLM을 쓸까? ###
*	지능형 모델 스위칭 (Model Switching): "비용이 너무 비싸네? Claude로 바꿔볼까?" 하는 고민이 코드 수정 없이 즉시 실행됩니다. 운영 환경에서 특정 모델 API가 터졌을 때 다른 모델로 자동 전환하는 'Fallbacks' 기능은 덤이죠.
*	비용과 예산 관리: 에이전트가 내 계좌를 털어먹지 않도록, API 키별로 예산 한도(Budget Manager)를 설정할 수 있습니다. 멍청한 에이전트가 무한 루프를 돌며 발생시키는 비용 사고를 방지하는 최후의 보루인 셈입니다.
*	통합 모니터링: 각 모델마다 파편화된 로그를 하나로 모아줍니다. 어떤 모델이 가장 삑사리(오류)를 많이 내는지 한눈에 파악할 수 있어, 시스템 개선의 우선순위를 정하기가 훨씬 쉬워집니다.


## EKS 배포하기 ##

### 1. EKS에 LiteLLM Gateway 배포 ###
```
apiVersion: apps/v1
kind: Deployment
metadata:
  name: litellm-gateway
spec:
  replicas: 2
  selector:
    matchLabels:
      app: litellm
  template:
    metadata:
      labels:
        app: litellm
    spec:
      containers:
      - name: litellm
        image: ghcr.io/berriai/litellm:main
        ports:
        - containerPort: 4000
        env:
        - name: LITELLM_MASTER_KEY
          value: "sk-my-super-secret-key"
        # AWS Bedrock 사용을 위한 IAM 설정이 EKS 노드에 필요합니다
        - name: AWS_REGION
          value: "us-east-1"
        - name: OPENAI_API_KEY
          value: "sk-..." 
        # 설정 파일 마운트 (아래 config.yaml 참고)
        volumeMounts:
        - name: config-volume
          mountPath: /app/config.yaml
          subPath: config.yaml
      volumes:
      - name: config-volume
        configMap:
          name: litellm-config
---
apiVersion: v1
kind: Service
metadata:
  name: litellm-service
spec:
  type: ClusterIP
  ports:
  - port: 4000
    targetPort: 4000
  selector:
    app: litellm
```

### 2. 모델 설정 (config.yaml - ConfigMap) ###
여기서 OpenAI, Bedrock, Open LLM(vLLM/Ollama 등)을 한 번에 정의합니다.
```
general_settings:
  master_key: "sk-my-super-secret-key"

model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY

  - model_name: claude-3-5
    litellm_params:
      model: bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0
      # AWS 환경변수 자동 인식

  - model_name: my-open-llm
    litellm_params:
      model: openai/meta-llama/llama-3-70b
      api_base: http://vllm-service.default.svc.cluster.local:8000/v1
      api_key: any-string
```
아래와 같이 컨피그맵을 생성한다.
```
kubectl create configmap litellm-config --from-file=config.yaml
```
