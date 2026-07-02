## 오픈소스 100B Omni-VLM 고성능 추론(지식 증류 데이터 생성) 파이프라인 ##

100B 모델은 FP16 기준 가중치만 약 200GB이고, 이미지/비디오/음성 텐서가 들어가면 컨텍스트가 폭발합니다.
싱글 노드(A100/H100 8장)에서 **vLLM 텐서 병렬화(TP=8)**로 엔진을 띄우고, CPU 워커로 멀티미디어를
디코딩해 밀어 넣는 구조가 효율적입니다. 이 문서는 그 파이프라인의 이론 → 구현 → 실전 체크리스트입니다.

### 1. 이론적 배경 (왜 이렇게 설계하는가) ###

#### 1.1 왜 텐서 병렬화(TP)인가 ###
100B 파라미터를 bf16로 올리면 가중치만 ~200GB. GPU 1장(80GB)에 안 들어갑니다. TP=8은 각 행렬 연산을
**8장에 행/열로 쪼개** 동시에 계산하고 all-reduce로 합칩니다. 노드 내부는 NVLink 대역폭이 커서 TP가
유리하고, **노드를 넘어가면** 통신 비용 때문에 파이프라인 병렬(PP)이나 다중 인스턴스가 더 낫습니다.
즉 "싱글 노드 8장 = TP=8"이 정석입니다.

#### 1.2 모델의 "능력"과 입력 모달리티는 일치해야 한다 (가장 중요) ####
증류 파이프라인에서 가장 흔한 치명적 실수는 **모델이 지원하지 않는 모달리티를 프롬프트에 넣는 것**입니다.

- `meta-llama/Llama-3.2-90B-Vision-Instruct`는 **이미지 전용**입니다. 오디오/비디오 입력을 받지 못하며
  `<|sound|>`, `<|video|>` 같은 토큰도 존재하지 않습니다. 넣으면 무시되거나 에러가 납니다.
- **이미지 + 오디오 + 비디오를 모두** 받는 진짜 "Omni" 오픈 모델은 Qwen2.5-Omni 계열이 대표적입니다
  (단, 100B급 오픈 Omni 모델은 현재 희소하므로 모델 선택 시 실제 지원 모달리티를 먼저 확인).
- 진짜 옴니 모델이 없다면 **모달리티별로 특화 모델을 분리 운용**(비전=Qwen2.5-VL, 오디오=Qwen2-Audio)
  하는 게 현실적입니다.

#### 1.3 vLLM의 `multi_modal_data`는 "경로"가 아니라 "디코딩된 객체" ####
vLLM은 파일 경로 문자열을 자동으로 열어주지 않습니다. 규약은 다음과 같습니다.

- `image`: `PIL.Image` 객체 (또는 리스트)
- `audio`: `(np.ndarray, sample_rate)` 튜플
- `video`: 프레임 배열(`np.ndarray` 시퀀스)

따라서 CPU 워커에서 **미리 디코딩**해 객체로 만들어 넣어야 합니다. 이게 프롬프트의 플레이스홀더
토큰(`<|image|>` 등) 개수와 1:1로 맞아야 합니다.

#### 1.4 증류용 샘플링 전략 ####
지식 증류(distillation)는 교사 모델의 "일관된" 지식을 뽑는 게 목적이므로 `temperature`를 낮게
(0.0~0.3) 둡니다. 다양성이 필요하면 프롬프트를 다변화하되 온도는 낮게 유지하는 편이 데이터 품질에
유리합니다.

### 2. 구현 코드 ###

원본의 핵심 수정: `json` import 추가, `multi_modal_data`를 디코딩된 객체로 로딩,
모델-모달리티 능력 일치, `limit_mm_per_prompt` 설정, 예외 처리 및 스트리밍 저장,
프롬프트는 프로세서 chat template로 생성 권장.

```python
import os
import json
import traceback
from pathlib import Path

from PIL import Image
import librosa                      # 오디오 디코딩 (pip install librosa)
from vllm import LLM, SamplingParams

# ------------------------------------------------------------------
# 0. 모델 선택: '실제로 오디오/비디오를 지원하는' 옴니 모델을 써야 한다.
#    - Llama-3.2-90B-Vision 은 '이미지 전용'이라 sound/video 입력 불가!
#    - 아래는 이미지+오디오+비디오를 지원하는 옴니 계열 예시.
# ------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-Omni-7B"   # 예시. 100B급이면 지원 모달리티를 반드시 확인
DATA_ROOT = Path("./data")
OUT_PATH = Path("omni_synthetic_dataset.jsonl")

# fork 기반 멀티프로세싱 워커에서 토크나이저 데드락 경고 회피
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("🚀 Omni-VLM 분산 추론 엔진 로딩 시작 (TP=8)...")
llm = LLM(
    model=MODEL_ID,
    tensor_parallel_size=8,                 # 싱글 노드 8장 = TP=8
    max_model_len=32768,                    # 비디오/오디오 토큰까지 고려해 넉넉히
    limit_mm_per_prompt={"image": 2, "audio": 1, "video": 1},  # 프롬프트당 모달 상한
    dtype="bfloat16",
    gpu_memory_utilization=0.90,
    trust_remote_code=True,
)

sampling_params = SamplingParams(
    temperature=0.2,   # 증류용 → 일관성 위해 낮게
    top_p=0.9,
    max_tokens=1536,
)


# ------------------------------------------------------------------
# 1. 멀티미디어를 '디코딩된 객체'로 로딩 (경로 문자열 그대로 넣으면 안 됨)
# ------------------------------------------------------------------
def load_multimodal(paths: dict) -> dict:
    """paths 예: {"image": ["a.jpg"], "audio": ["b.wav"]}"""
    data = {}
    if "image" in paths:
        data["image"] = [Image.open(DATA_ROOT / p).convert("RGB") for p in paths["image"]]
    if "audio" in paths:
        # vLLM audio 규약: (np.ndarray, sample_rate) 튜플
        data["audio"] = [librosa.load(DATA_ROOT / p, sr=16000) for p in paths["audio"]]
    if "video" in paths:
        # 프레임 배열로 디코딩 (예: decord/opencv). 여기선 헬퍼로 위임
        data["video"] = [load_video_frames(DATA_ROOT / p) for p in paths["video"]]
    return data


def load_video_frames(path):
    """decord 등으로 균일 샘플링한 프레임 배열을 반환 (구현은 환경에 맞게)."""
    from decord import VideoReader  # pip install decord
    vr = VideoReader(str(path))
    # 예: 균일하게 16프레임 샘플링
    idx = [int(i) for i in range(0, len(vr), max(1, len(vr) // 16))][:16]
    return vr.get_batch(idx).asnumpy()


# ------------------------------------------------------------------
# 2. 입력 스펙: 프롬프트 + 모달 경로 (플레이스홀더 개수는 모달 개수와 1:1)
#    실제 플레이스홀더 토큰은 모델 프로세서마다 다르므로 apply_chat_template 권장.
# ------------------------------------------------------------------
raw_specs = [
    {
        "prompt": "<|image|><|audio|>이 이미지의 시각 요소와 오디오의 연관성을 분석하고, "
                  "1B 모델 학습용 고품질 VQA 세트를 작성해라.",
        "mm_paths": {"image": ["sample_img.jpg"], "audio": ["sample_audio.wav"]},
    },
    {
        "prompt": "<|video|>이 비디오의 프레임별 인과관계를 요약하고, "
                  "타임스탬프별 액션-답변 데이터셋을 추출해라.",
        "mm_paths": {"video": ["sample_video.mp4"]},
    },
]

# vLLM 입력 형태로 변환 (경로 → 디코딩 객체)
vllm_inputs, kept_specs = [], []
for spec in raw_specs:
    try:
        vllm_inputs.append({
            "prompt": spec["prompt"],
            "multi_modal_data": load_multimodal(spec["mm_paths"]),
        })
        kept_specs.append(spec)
    except Exception:
        print(f"⚠️  디코딩 실패, 스킵: {spec['mm_paths']}")
        traceback.print_exc()


# ------------------------------------------------------------------
# 3. 배치 추론 + 스트리밍 저장 (대량이면 청크 단위로 generate 권장)
# ------------------------------------------------------------------
print("⚡ 멀티모달 배치 추론 가동 중...")
outputs = llm.generate(vllm_inputs, sampling_params)

with OUT_PATH.open("w", encoding="utf-8") as f:
    for idx, output in enumerate(outputs):
        if not output.outputs:
            continue
        record = {
            "id": f"omni-{idx:06d}",
            "teacher_prompt": kept_specs[idx]["prompt"],
            "mm_paths": kept_specs[idx]["mm_paths"],
            "distilled_response": output.outputs[0].text,
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"✅ 지식 증류 데이터셋 추출 완료 → {OUT_PATH}")
```


### 3. 실전 투입 전 체크리스트 ###

1. **모델 능력 = 입력 모달리티 일치**: 가장 먼저 확인. 비전 전용 모델에 오디오/비디오를 넣지 말 것.
   진짜 옴니가 없으면 모달리티별 특화 모델로 분리 운용.
2. **`multi_modal_data`는 디코딩 객체**: image=PIL, audio=(ndarray, sr), video=프레임 배열.
   경로 문자열 그대로 넣으면 동작하지 않음.
3. **플레이스홀더 정합성**: 프롬프트의 `<|image|>` 등 개수 = 실제 모달 객체 개수. 가능하면
   `processor.apply_chat_template(...)`로 프롬프트를 생성해 토큰 규약을 자동으로 맞출 것.
4. **`limit_mm_per_prompt` 설정**: 프롬프트당 이미지/오디오/비디오 상한을 명시하지 않으면 다중 모달
   입력에서 에러가 날 수 있음.
5. **`max_model_len` 여유**: 비디오/오디오는 토큰 수가 폭증. 8192는 대개 부족하니 32k 이상 고려하되
   KV 캐시 메모리와 트레이드오프 확인.
6. **CPU 디코딩 병목 분리**: 이미지/비디오/오디오 디코딩은 Ray Data나 멀티프로세싱 워커로 GPU 추론과
   분리해 파이프라이닝(디코딩 ↔ 추론 오버랩).
7. **대량 처리 시 청크 배칭**: 전체를 한 번에 메모리에 올리지 말고, 수백~수천 건 단위로 나눠
   `generate` 호출하고 결과를 즉시 jsonl에 append(스트리밍 저장).
8. **예외 격리 + 재시도**: 손상 파일 하나가 전체 배치를 죽이지 않도록 per-sample try/except,
   실패 목록 별도 로깅 후 재처리.
9. **증류 데이터 품질 관리**: 낮은 temperature + 프롬프트 다변화. 생성 후 길이/포맷/중복 필터,
   가능하면 규칙 기반 또는 소형 검증 모델로 자동 검수.
10. **재현성 메타데이터**: 각 레코드에 모델 ID, 샘플링 파라미터, 원본 미디어 경로/해시를 남겨
    이후 1B 학습 데이터의 출처 추적을 보장.
