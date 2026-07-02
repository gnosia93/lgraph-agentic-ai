## DeepSpeed ZeRO-3 + Offload 기반 1B Omni-VLM 분산 증류 학습 ##

비디오/오디오 텐서는 이미지보다 훨씬 커서 분산 훈련 시 메모리 고갈이 쉽게 발생합니다.
Slurm-on-AWS / training-on-EKS 클러스터에서 ZeRO-Stage 3 + CPU 오프로드로 이를 완화하는
파이프라인에 대한 설명입니다. 

### 1. 이론적 배경 ###

#### 1.1 ZeRO(Zero Redundancy Optimizer) 3단계 ####
데이터 병렬(DDP)은 모든 GPU가 **모델 가중치·그래디언트·옵티마이저 상태를 통째로 중복 보유**합니다.
ZeRO는 이 셋을 GPU들에 **분할(partition)**해서 중복을 없앱니다.

- **Stage 1**: 옵티마이저 상태(Adam의 m, v 등)를 분할 — 메모리 절감 최다 구간.
- **Stage 2**: + 그래디언트도 분할.
- **Stage 3**: + **모델 파라미터 자체**를 분할. forward/backward 시 필요한 파라미터만 all-gather로
  잠깐 모았다가 버립니다. 통신량이 늘지만 단일 GPU가 담을 수 없는 큰 모델을 올릴 수 있습니다.

> ⚠️ 참고: 학생 모델이 **1B에 불과**하다면 ZeRO-3 + 오프로드는 통신·오프로드 오버헤드 때문에
> 오히려 느릴 수 있습니다. 1B급은 **DDP 또는 ZeRO-2**가 보통 더 빠릅니다. ZeRO-3는 교사(100B)
> 학습이나 초장문 컨텍스트에서 진가를 발휘합니다. 규모에 맞게 선택하세요.

#### 1.2 오프로드(Offload) ####
`offload_optimizer(device="cpu")`는 옵티마이저 상태를 CPU RAM으로 내려 GPU VRAM을 확보합니다.
`pin_memory=true`는 고정 메모리를 써서 CPU↔GPU 전송을 빠르게 합니다. 더 큰 모델이면
`offload_param`까지 CPU/NVMe로 내릴 수 있지만, 매 스텝 전송 비용으로 학습이 느려지는 트레이드오프가
있습니다.

#### 1.3 활성화 메모리와 Gradient Checkpointing (여기서 가장 중요) ####
비디오(1500+ 프레임 토큰)·오디오(1500 프레임)가 시퀀스에 들어가면 **활성화(activation) 메모리**가
시퀀스 길이에 비례해 폭증합니다. ZeRO는 파라미터/옵티마이저 메모리를 줄여줄 뿐, **활성화 메모리는
줄여주지 않습니다.** 따라서 긴 멀티모달 시퀀스에서는 **gradient(activation) checkpointing**으로
중간 활성화를 버렸다가 backward에서 재계산하는 기법이 사실상 필수입니다. (원본 코드에 누락)

#### 1.4 bf16을 쓰는 이유 ####
bf16은 fp16과 지수부 비트가 같아(8bit) 큰 dynamic range를 가져 **loss scaling 없이도 안정적**입니다.
A100/H100에서 표준으로 권장됩니다.

#### 1.5 ZeRO-3에서의 체크포인트 저장 ####
ZeRO-3는 파라미터가 샤딩돼 있어 일반 `torch.save(model.state_dict())`로는 온전한 가중치를 못 얻습니다.
방법은 둘 중 하나입니다.

- `model_engine.save_checkpoint(dir, tag)` (모든 랭크가 호출) → 이후 `zero_to_fp32.py`로 병합, 또는
- config에 `stage3_gather_16bit_weights_on_model_save: true`를 켜고
  `model_engine.save_16bit_model(dir)`로 16bit 통합 가중치 저장.

원본의 `save_1d_checkpoint`는 **존재하지 않는 API**입니다.


### 2. DeepSpeed 설정 (`deepspeed_omni_config.json`) ###

원본 대비 수정: 마이크로 배치 명시(“둘 다 auto”는 추론 불가), 16bit 통합 저장 플래그 추가,
활성화 체크포인팅 관련 주석.

```json
{
  "train_micro_batch_size_per_gpu": 2,
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": 1.0,
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": { "device": "cpu", "pin_memory": true },
    "offload_param": { "device": "none" },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_prefetch_bucket_size": 5e7,
    "stage3_param_persistence_threshold": 1e6,
    "stage3_gather_16bit_weights_on_model_save": true
  }
}
```

- `train_micro_batch_size_per_gpu`는 DataLoader의 `batch_size`와 반드시 일치시킵니다.
- `train_batch_size`를 굳이 안 넣고 `gradient_accumulation_steps: "auto"`로 두면
  `micro_batch × world_size × accum`으로 자동 계산됩니다.


### 3. 학습 스크립트 (`train_omni_vlm.py`) ###

핵심 수정: 분산 초기화 추가, **글로벌 RANK**로 샘플러 구성, `None` 모달리티를 견디는 커스텀
`collate_fn`, 백본 프리징(커넥터만 학습할 경우), 올바른 체크포인트 저장, gradient checkpointing.

```python
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import deepspeed

from student_model import StudentOmniVLM  # Module 2의 수정판 모델

# 1. 분산 랭크: 멀티노드에서는 '글로벌 RANK'로 샘플러를 나눠야 한다.
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
GLOBAL_RANK = int(os.environ.get("RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))

torch.cuda.set_device(LOCAL_RANK)
deepspeed.init_distributed()  # process group 초기화 (샘플러/통신 전 필수)


# 2. 합성 데이터셋 로더
class OmniDistillDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            self.samples = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # [실전 팁] 인코더를 통과한 임베딩을 사전 계산(.pt)해 두면 I/O 병목이 크게 준다.
        # 여기서는 인코더 출력 스펙에 맞춘 가상 임베딩을 생성.
        item = {
            "image_embeds": torch.randn(64, 1152),    # SigLIP
            "audio_embeds": torch.randn(1500, 1280),  # Whisper
            "input_ids": torch.randint(1, 10000, (512,)),
        }
        item["labels"] = item["input_ids"].clone()
        # 비디오가 없는 샘플은 키를 아예 넣지 않는다 (None 대신 '부재'로 표현)
        return item


# 3. None/부재 모달리티를 견디는 커스텀 collate
#    (배치 내 이미지/오디오 유무가 동일하다고 가정. 혼합 시엔 per-sample 처리 필요)
def omni_collate(batch):
    out = {}
    for key in ("image_embeds", "audio_embeds", "video_embeds"):
        present = [b[key] for b in batch if b.get(key) is not None]
        out[key] = torch.stack(present, dim=0) if len(present) == len(batch) else None
    out["input_ids"] = torch.stack([b["input_ids"] for b in batch], dim=0)
    out["labels"] = torch.stack([b["labels"] for b in batch], dim=0)
    return out


def main():
    if GLOBAL_RANK == 0:
        print("📦 멀티모달 증류 데이터셋 로딩...")

    dataset = OmniDistillDataset("./omni_synthetic_dataset.jsonl")
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=WORLD_SIZE, rank=GLOBAL_RANK, shuffle=True
    )
    data_loader = DataLoader(
        dataset,
        batch_size=2,              # config의 train_micro_batch_size_per_gpu와 일치
        sampler=sampler,
        pin_memory=True,
        num_workers=4,
        collate_fn=omni_collate,
    )

    # 4. 학생 모델 + (선택) 백본 프리징: 커넥터만 학습하려면 LLM을 freeze
    model = StudentOmniVLM()
    for p in model.llm.parameters():
        p.requires_grad = False          # 커넥터(프로젝터)만 학습하려면 활성화
    # 큰 활성화 메모리 대비: gradient checkpointing
    if hasattr(model.llm, "gradient_checkpointing_enable"):
        model.llm.gradient_checkpointing_enable()
        model.llm.config.use_cache = False  # checkpointing과 KV캐시는 함께 못 씀

    # 5. DeepSpeed 엔진 초기화
    if GLOBAL_RANK == 0:
        print("⚡ DeepSpeed Engine 초기화...")
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        config="deepspeed_omni_config.json",
    )
    device = model_engine.device

    # 6. 훈련 루프
    model_engine.train()
    for epoch in range(3):
        sampler.set_epoch(epoch)  # 에폭마다 셔플 시드 갱신 (필수)
        for step, batch in enumerate(data_loader):
            image_embeds = batch["image_embeds"].to(device, dtype=torch.bfloat16, non_blocking=True) \
                if batch["image_embeds"] is not None else None
            audio_embeds = batch["audio_embeds"].to(device, dtype=torch.bfloat16, non_blocking=True) \
                if batch["audio_embeds"] is not None else None
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            attention_mask = torch.ones_like(input_ids)

            outputs = model_engine(
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_embeds=image_embeds,
                audio_embeds=audio_embeds,
                labels=labels,
            )
            loss = outputs.loss

            model_engine.backward(loss)
            model_engine.step()

            if GLOBAL_RANK == 0 and step % 10 == 0:
                print(f"📈 [EPOCH {epoch}] [STEP {step}] Loss: {loss.item():.4f}")

    # 7. 체크포인트 저장
    #    (a) 재개용 샤딩 체크포인트: 모든 랭크가 호출해야 함
    model_engine.save_checkpoint("./checkpoints/omni_vlm_1b")
    #    (b) 배포용 16bit 통합 가중치 (config에 gather 플래그 필요)
    model_engine.save_16bit_model("./checkpoints/omni_vlm_1b_final")
    if GLOBAL_RANK == 0:
        print("💾 체크포인트 저장 완료.")


if __name__ == "__main__":
    main()
```


### 4. 실전 체크리스트 ###

1. **모델 규모에 맞는 ZeRO 단계**: 1B 학생이면 ZeRO-2 또는 DDP가 대개 더 빠름. ZeRO-3 + 오프로드는
   통신/전송 오버헤드가 커서 소형 모델엔 과함.
2. **글로벌 RANK로 샘플러 분할**: 멀티노드에서 `DistributedSampler(rank=...)`에 `LOCAL_RANK`를 쓰면
   노드 간 데이터가 중복됨. 반드시 글로벌 `RANK` 사용.
3. **분산 초기화 순서**: `torch.cuda.set_device` → `deepspeed.init_distributed()` → 샘플러 생성.
4. **`None` 모달리티 collate**: 기본 `default_collate`는 `None`을 못 묶음. 커스텀 `collate_fn`으로
   부재 모달리티를 안전 처리(키 제거 또는 배치 단위 정합성 보장).
5. **Gradient(activation) checkpointing**: 긴 비디오/오디오 시퀀스에서 활성화 메모리 폭증 방지에
   사실상 필수. 켜면 `use_cache=False`도 함께.
6. **마이크로 배치 정합성**: DataLoader `batch_size` = config `train_micro_batch_size_per_gpu`.
   둘 다 `"auto"`로 두면 DeepSpeed가 추론 불가로 에러.
7. **커넥터만 학습 시 백본 freeze**: 코드에서 실제로 `requires_grad=False`를 걸어야 함.
   주석만 "커넥터 타겟팅"이라 적고 안 얼리면 전체가 학습됨.
8. **체크포인트 API**: `save_1d_checkpoint`는 없음. `save_checkpoint`(재개용) +
   `save_16bit_model`(배포용, gather 플래그 필요) 또는 `zero_to_fp32.py` 변환 사용.
9. **사전 임베딩 캐싱**: 인코더 forward를 매 스텝 돌리지 말고 `.pt`로 구워 S3/Lustre에서 로딩 →
   I/O·연산 병목 대폭 감소.
10. **가변 길이 처리**: 실제 데이터는 시퀀스 길이가 제각각. 학생 모델의 융합 후 패딩과 별개로,
    DataLoader 단계에서도 길이 정렬/버킷팅으로 패딩 낭비를 줄이면 처리량이 오름.
