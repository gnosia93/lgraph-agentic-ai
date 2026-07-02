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
