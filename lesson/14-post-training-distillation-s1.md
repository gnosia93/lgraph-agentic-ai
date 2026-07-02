
100B급 대형 모델(교사)로부터 수십만 장의 이미지에 대한 캡션 및 VQA(질의응답) 데이터를 추출할 때의 병목은 **'네트워크 I/O'**와 **'LLM 스루풋(Throughput)'**입니다.
여기서는 **AWS Bedrock의 비동기 일괄 처리(Batch Inference)**를 활용한 파이프라인 아키텍처를 구현합니다. 실시간 API 대비 비용이 50% 저렴하며, API Rate Limit(호출 제한) 스트레스 없이 S3 버킷을 통해 대량의 데이터를 가장 안전하게 밀어 넣을 수 있는 엔터프라이즈 표준 방식입니다.


### 1) 데이터 파이프라인 아키텍처 흐름 ###

[Raw Images 인덱싱] ➡️ [S3 Manifest JSONL 생성] ➡️ [Bedrock Batch Job 생성] ➡️ [비동기 100B 추론] ➡️ [S3 결과 적재] ➡️ [1B 학습용 전처리]

### 2) Bedrock Batch Inference 입력용 Manifest 생성 및 Job 요청 스크립트 ###
```
import json
import boto3
import time

# 1. AWS Bedrock 초기화
bedrock_client = boto3.client(service_name="bedrock", region_name="us-east-1")
s3_client = boto3.client(service_name="s3", region_name="us-east-1")

BUCKET_NAME = "your-ai-distillation-data-bucket"
MANIFEST_KEY = "inputs/bedrock_batch_input.jsonl"

# 2. 100B 모델(Claude 3.5 Sonnet)에게 보낼 멀티모달 프롬프트 Manifest 생성
# 실제 환경에서는 대량의 S3 이미지 경로를 루프 돌며 JSONL 파일로 빌드합니다.
raw_samples = [
    {
        "image_url": f"s3://{BUCKET_NAME}/raw_images/sample_001.jpg",
        "image_format": "jpeg"
    },
    {
        "image_url": f"s3://{BUCKET_NAME}/raw_images/sample_002.jpg",
        "image_format": "jpeg"
    }
]

print("🔄 Bedrock Batch Inference용 JSONL 입력 파일 작성 중...")
with open("bedrock_batch_input.jsonl", "w", encoding="utf-8") as f:
    for idx, sample in enumerate(raw_samples):
        # Bedrock Batch가 요구하는 개별 라인 포맷 정의
        batch_record = {
            "recordId": f"req-{idx:06d}",
            "modelId": "anthropic.claude-3-5-sonnet-20240620-v1:0",
            "modelInput": {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "temperature": 0.2, # 데이터 합성의 일관성을 위해 낮게 책정
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "s3",
                                    "bucket": BUCKET_NAME,
                                    "key": sample["image_url"].split(f"{BUCKET_NAME}/")[1]
                                }
                            },
                            {
                                "type": "text",
                                "text": "이 이미지의 세부 객체, 배경, 인과관계를 단계별로 분석하여 아주 상세한 고품질 캡션을 작성해줘. 그리고 이 이미지를 기반으로 추론할 수 있는 복잡한 수준의 시각적 질의응답(VQA) 세트를 '질문:'과 '답변:' 형태로 3개 만들어줘."
                            }
                        ]
                    }
                ]
            }
        }
        f.write(json.dumps(batch_record, ensure_ascii=False) + "\n")

# S3에 Manifest 업로드
s3_client.upload_file("bedrock_batch_input.jsonl", BUCKET_NAME, MANIFEST_KEY)
print(f"✅ Manifest 업로드 완료: s3://{BUCKET_NAME}/{MANIFEST_KEY}")

# 3. Bedrock 비동기 Batch Job 생성 호출
print("🚀 Bedrock 100B Model Batch Job 요청 중...")
response = bedrock_client.create_model_invocation_job(
    jobName="vlm-100b-data-generation-job",
    modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
    inputDataConfig={"s3InputDataConfig": {"s3Uri": f"s3://{BUCKET_NAME}/{MANIFEST_KEY}"}},
    outputDataConfig={"s3OutputDataConfig": {"s3Uri": f"s3://{BUCKET_NAME}/outputs/"}},
    roleArn="arn:aws:iam::your-account-id:role/service-role/AmazonBedrockExecutionRoleForJobs"
)

job_arn = response["jobArn"]
print(f"✅ Job이 성공적으로 생성되었습니다. Job ARN: {job_arn}")
```
