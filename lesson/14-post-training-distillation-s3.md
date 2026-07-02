비디오와 오디오 데이터는 이미지보다 텐서 크기가 압도적으로 커서 분산 훈련 시 무조건 자원 고갈이 일어납니다. 강사님의 slurm-on-aws나 training-on-eks 클러스터에서 가속기 스왑을 원활하게 굴릴 수 있는 DeepSpeed ZeRO-Stage 3 + 가속기/CPU 메모리 오프로드(Offload) 조합 설정입니다


deepspeed_omni_config.json
```
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "none" 
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_prefetch_bucket_size": 5e7,
    "stage3_param_persistence_threshold": 1e6
  },
  "bf16": {
    "enabled": true
  },
  "gradient_clipping": 1.0
}
```

### 1B Omni-VLM 실전 분산 훈련 스크립트 (train_omni_vlm.py) ###
```
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import deepspeed

# Module 2에서 정의한 커스텀 1B 멀티헤드 모델 구조 임포트
from student_model import StudentOmniVLM1B

# 1. 분산 환경을 위한 로컬 랭크(Rank) 파싱 (DeepSpeed가 자동으로 주입)
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
torch.cuda.set_device(LOCAL_RANK)

# 2. 100B 오프라인 파이프라인이 생성한 멀티모달 합성 데이터셋 로더 정의
class OmniDistillDataset(Dataset):
    def __init__(self, data_path):
        self.samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # [실전 인프라 팁] 
        # 멀티모달 학습 시 실제 이미지/오디오 파싱은 전처리(Pre-tokenize) 단계에서 
        # 고정 차원의 가속기 텐서(Embedding)로 구워두고 S3/Lustre에서 텐서 파일(.pt)로 읽는 것이 I/O 병목 방지에 압도적입니다.
        # 여기서는 인코더를 통과한 가상의 임베딩 텐서를 예시로 생성합니다.
        image_embeds = torch.randn(1, 64, 1152)  # SigLIP 아웃풋 스펙
        audio_embeds = torch.randn(1, 1500, 1280) # Whisper 아웃풋 스펙
        video_embeds = None                       # 샘플 데이터에 따라 선택적 매핑
        
        # 100B 교사가 만들어준 정답 텍스트 토크나이징 (예시 가상 토큰)
        input_ids = torch.randint(1, 10000, (1, 512))
        labels = input_ids.clone() # Autoregressive 언어 모델 학습용 레이블 생성
        
        return {
            "image_embeds": image_embeds.squeeze(0),
            "audio_embeds": audio_embeds.squeeze(0),
            "video_embeds": video_embeds, # 필요 시 활성화
            "input_ids": input_ids.squeeze(0),
            "labels": labels.squeeze(0)
        }

def main():
    # 3. 분산 학습을 위한 데이터셋 및 분산 샘플러 배치
    print(f"📦 [RANK {LOCAL_RANK}] 멀티모달 증류 데이터셋 로딩 중...")
    dataset = OmniDistillDataset("./omni_synthetic_dataset.jsonl")
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=WORLD_SIZE, rank=LOCAL_RANK, shuffle=True
    )
    
    # 텐서 크기가 큰 멀티모달 특성을 고려해 pin_memory 활성화
    data_loader = DataLoader(dataset, batch_size=2, sampler=sampler, pin_memory=True)

    # 4. 1B 학생 모델 인스턴스화
    model = StudentOmniVLM1B()

    # 5. DeepSpeed 엔진 초기화 (Stage 3 텐서 파티셔닝 가동)
    # 쉘 스크립트에서 넘겨받은 deepspeed_omni_config.json을 자동으로 바인딩합니다.
    print(f"⚡ [RANK {LOCAL_RANK}] DeepSpeed Engine 초기화 중...")
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad], # 학습 활성화된 커넥터 레이어 타겟팅
        config="deepspeed_omni_config.json"
    )

    # 6. 실전 훈련 루프 가동
    model_engine.train()
    for epoch in range(3):
        sampler.set_epoch(epoch)
        for step, batch in enumerate(data_loader):
            
            # 장치(GPU)로 텐서 이관
            image_embeds = batch["image_embeds"].to(model_engine.local_rank, dtype=torch.bfloat16)
            audio_embeds = batch["audio_embeds"].to(model_engine.local_rank, dtype=torch.bfloat16)
            input_ids = batch["input_ids"].to(model_engine.local_rank)
            labels = batch["labels"].to(model_engine.local_rank)

            # 포워드 연산 (Forward Pass)
            outputs = model_engine(
                image_embeds=image_embeds,
                audio_embeds=audio_embeds,
                video_embeds=None,
                input_ids=input_ids,
                labels=labels
            )
            
            loss = outputs.loss

            # 백워드 연산 (DeepSpeed 전용 배포형 역전파 엔진)
            model_engine.backward(loss)
            
            # 가중치 업데이트 및 동기화 (Gradient Step)
            model_engine.step()

            if LOCAL_RANK == 0 and step % 10 == 0:
                print(f"📈 [EPOCH {epoch}] [STEP {step}] Loss: {loss.item():.4f}")

    # 7. 안전한 체크포인트 저장 (ZeRO-Stage 3 분산 가중치 병합 저장)
    print("💾 훈련 완료. 모델 가중치를 영구 저장합니다.")
    model_engine.save_1d_checkpoint("./checkpoints/omni_vlm_1b_final")

if __name__ == "__main__":
    main()
```

