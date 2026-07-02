## 오픈소스 100B Omni-VLM 고성능 추론 파이프라인 아키텍처 ##

100B 모델은 가중치 용량만 FP16 기준 약 200GB에 달하며, 이미지/비디오/음성 텐서가 들어가면 컨텍스트 윈도우가 폭발합니다. 따라서 싱글 노드(A100/H100 8장) 환경에서 **vLLM의 텐서 병렬화(TP=8)**를 활용해 분산 추론 엔진을 띄우고, 파이썬 멀티프로세싱으로 데이터를 밀어 넣는 구조가 가장 효율적입니다.

### 1) 인프라 토폴로지 및 데이터 흐름 ###
* S3/Lustre 스토리지: 이미지, 비디오(.mp4), 음성(.wav) 원천 데이터 저장
* Ray Data / Multiprocessing 워커: 멀티미디어 데이터를 텐서로 디코딩 (CPU 병목 방지)
* vLLM 대형 추론 엔진: TP=8로 100B 모델을 GPU 8장에 쪼개 올려 실시간/배치 분산 추론 가동
  
### 2) 100B Omni-VLM 분산 배치 추론 스크립트 (extract_omni_data.py) ###
```
import os
from vllm import LLM, SamplingParams

# 1. 분산 환경 변수 및 vLLM 최적화 설정
# 100B 모델을 GPU 8장에 분산하여 로딩 (Tensor Parallelism = 8)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

print("🚀 오픈소스 100B Omni-VLM 분산 추론 엔진 로딩 시작 (TP=8)...")
llm = LLM(
    model="meta-llama/Llama-3.2-90B-Vision-Instruct", # 또는 최신 100B급 오픈소스 Omni 모델
    tensor_parallel_size=8,                            # HGX A100/H100 8장 전용 세팅
    max_model_len=8192,                                # 멀티모달 컨텍스트 확보
    trust_remote_code=True
)

# 2. 멀티모달 합성 데이터 생성을 위한 정밀한 샘플링 가이드
sampling_params = SamplingParams(
    temperature=0.2,   # 지식 증류용 데이터이므로 일관성을 위해 낮게 세팅
    max_tokens=1540,
    top_p=0.9
)

# 3. 이미지, 비디오, 음성이 융합된 멀티모달 입력 예시 구조화
# 실제 환경에서는 루프를 돌며 S3나 FSx for Lustre의 로컬 경로를 받아옵니다.
omni_inputs = [
    {
        "prompt": "<|image|><|sound|>이 이미지의 시각적 요소와 오디오 가이드 음성의 연관성을 분석하고, 1B 모델 학습을 위한 고품질 VQA 세트를 작성해라.",
        "multi_modal_data": {
            "image": "./data/sample_img.jpg",
            "sound": "./data/sample_audio.wav"
        }
    },
    {
        "prompt": "<|video|>이 10초짜리 비디오의 프레임별 인과관계를 요약하고, 타임스탬프별 액션-답변 데이터셋을 추출해라.",
        "multi_modal_data": {
            "video": "./data/sample_video.mp4"
        }
    }
]

# 4. 고성능 일괄 배칭(Batching) 추론 실행
print("⚡ 멀티모달 배치 추론 가동 중...")
outputs = llm.generate(omni_inputs, sampling_params)

# 5. 추출된 데이터를 1B 모델의 '골든 데이터셋'으로 저장
with open("omni_synthetic_dataset.jsonl", "w", encoding="utf-8") as f:
    for idx, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        result_node = {
            "id": f"omni-{idx:06d}",
            "teacher_prompt": omni_inputs[idx]["prompt"],
            "distilled_response": generated_text
        }
        f.write(json.dumps(result_node, ensure_ascii=False) + "\n")

print("✅ 100B Omni 모델로부터 지식 증류 데이터셋 추출 완료.")
```



