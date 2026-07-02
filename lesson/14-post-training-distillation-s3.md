멀티모달 데이터는 고해상도 이미지 행렬 연산 때문에 가속기 메모리(VRAM) 폭발(OOM)이 빈번하게 터집니다.
노드 수십 대를 엮어도 메모리를 안정적으로 쪼개 쓰며 성능을 최대로 끌어올릴 수 있는 DeepSpeed ZeRO-Stage 2 기반 가속화 설정 파일과 PyTorch Distributed 분산 훈련 실행 스크립트입니다.

### 1) deepspeed_config.json (메모리 최적화 튜닝 커스텀) ###
```
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e7,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e7,
    "contiguous_gradients": true
  },
  "fp16": {
    "enabled": true
  },
  "gradient_clipping": 1.0,
  "steps_per_print": 10
}
```

### 2) run_distributed_train.sh (멀티노드 NCCL 최적화 구동 쉘) ###
EFA 및 가속기 토폴로지를 완전히 쥐어짜는 배치 스크립트 예시입니다. Slurm의 #SBATCH나 EKS의 TrainingOperator 파드 엔트리포인트에 직접 주입할 수 있습니다.
```
#!/bin/bash

# 1. 인프라 환경 변수 세팅 (멀티노드 마스터 주소 지정)
export MASTER_ADDR=${MASTER_ADDR:-"10.0.1.50"} # 마스터 노드 프라이빗 IP
export MASTER_PORT=${MASTER_PORT:-"9901"}
export NNODES=${WORLD_SIZE:-2}                 # 총 노드(서버) 수
export NODE_RANK=${RANK:-0}                    # 현재 노드의 인덱스 번호
export GPUS_PER_NODE=8                         # 노드당 인스턴스 칩 개수 (ex. A100 8장)

# 2. 멀티모달 통신 오버헤드 최소화를 위한 NCCL 가속 바인딩 옵션
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0                       # AWS 환경 시 EFA 바인딩 활성화
export NCCL_NET_GDR_LEVEL=5                    # GPU Direct RDMA 최대 활성화

echo "🚀 [NODE RANK: ${NODE_RANK}] Distributed Training 가동 커맨드를 전파합니다."

# 3. deepspeed 러너 가동 명령어 인터페이스
deepspeed --num_nodes=${NNODES} \
          --num_gpus=${GPUS_PER_NODE} \
          --master_addr=${MASTER_ADDR} \
          --master_port=${MASTER_PORT} \
          train_vlm.py \
          --deepspeed deepspeed_config.json \
          --model_path "./my_custom_vlm_1b" \
          --data_path "s3://your-ai-distillation-data-bucket/outputs/" \
          --epochs 3 \
          --per_device_train_batch_size 4 \
          --gradient_accumulation_steps 2 \
          --learning_rate 2e-5
```
