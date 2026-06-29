## Post Training ##
Post-training은 대규모 텍스트로 언어 능력을 학습하는 사전 학습(Pre-training)이 완료된 Base Model에 추가로 적용하는 모든 기법을 말한다. 크게 두 갈래로 나뉘는데, 첫 번째는 품질 개선으로 SFT(지시를 따르도록 학습), DPO(응답 품질 및 안전성 정렬), RLHF(사람 피드백 기반 강화학습) 등 추가 학습을 통해 모델의 응답 능력을 향상시키는 방법이다. 두 번째는 효율 최적화로, Quantization(BF16에서 INT4로 변환하여 메모리 절감), Pruning(불필요한 가중치를 제거하여 속도 향상), Distillation(큰 모델의 지식을 작은 모델로 전달) 등 별도의 학습 없이도 모델을 경량화하고 추론 성능을 개선하는 방법이다.
```
Pre-training → SFT → DPO → Quantization/Pruning → 배포
              └──────── Post-training ────────┘
```

### SFT vs DPO 차이 ###
* SFT:  "이 응답을 따라해" (정답 모방)
* DPO:  "이 응답이 저것보다 나아" (선호도 학습)
  
보통 SFT 먼저 하고, 그 위에 DPO를 추가로 하는 게 일반적이다. 
```
Pre-training → SFT (기본 능력 학습) → DPO (응답 품질 개선)
```
SFT는 "뭘 답해야 하는지" 가르치는 거고, DPO는 "어떻게 답해야 더 좋은지" 가르치는 것으로, 둘 다 instruction 형태의 데이터를 쓰긴 하지만, 학습 목적과 loss 함수가 다르다.

### SFT (인스트럭션 튜닝) ###
* https://raw.githubusercontent.com/gnosia93/eks-agentic-ai/refs/heads/main/code/qwen_finetune_inst.py

### DPO (선호도 정렬 - Alignment) ###
* https://raw.githubusercontent.com/gnosia93/eks-agentic-ai/refs/heads/main/code/qwen_finetune_dpo.py
  
beta=0.1이 핵심 하이퍼파라미터인데, 작을수록 chosen/rejected 차이를 크게 반영하고, 클수록 보수적으로 학습한다(0.1~0.5 사이에서 실험)

### Pruning (가지치기) ###
Pruning은 크게 세 가지 방식이 있다. 비구조적(Unstructured) pruning은 개별 가중치를 0으로 만드는 방식으로, 50% 스파시티를 적용하면 이론적으로 메모리를 절반으로 줄일 수 있지만, 실제 속도 향상을 얻으려면 sparse 연산을 지원하는 하드웨어가 필요하다. 반구조적(Semi-structured) pruning은 2:4 패턴(4개 가중치 중 2개를 0으로)을 적용하는 방식으로, NVIDIA Ampere 이상 GPU의 Sparse Tensor Core가 이 패턴을 하드웨어 레벨에서 인식하여 실제 추론 속도를 약 2배 향상시킬 수 있다. 다만 비구조적/반구조적 모두 가중치를 0으로 만들 뿐 텐서 shape은 유지되므로 모델 파일 크기 자체는 줄어들지 않는다. 구조적(Structured) pruning은 레이어, 어텐션 헤드, 뉴런 단위로 통째로 제거하는 방식으로, 파라미터 수와 모델 크기가 실제로 줄어들지만 세 가지 중 품질 손실이 가장 크다

#### Wanda 로 Purning 하기 ####
```
# 1. Wanda 설치
git clone https://github.com/locuslab/wanda.git
cd wanda
pip install -e .

# 2. 비구조적 pruning (50% sparsity)
python main.py \
  --model Qwen/Qwen3.5-27B \
  --prune_method wanda \
  --sparsity_ratio 0.5 \
  --save out/qwen-pruned-50 \
  --save_model out/qwen-pruned-50

# 3. 2:4 반구조적 pruning
python main.py \
  --model Qwen/Qwen3.5-27B \
  --prune_method wanda \
  --sparsity_ratio 0.5 \
  --prune_n 2 \
  --prune_m 4 \
  --save out/qwen-pruned-2-4 \
  --save_model out/qwen-pruned-2-4
```

* 저장된 모델 로드:
```
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("out/qwen-pruned-2-4")
tokenizer = AutoTokenizer.from_pretrained("out/qwen-pruned-2-4")
```
저장된 모델 파일 크기는 원본과 같다. 이는 0인 가중치도 그대로 저장되기 때문인데, 파일 크기까지 줄이려면 pruning 후 양자화를 추가로 적용해야 한다.

### Distillation (지식 증류) ###
* https://raw.githubusercontent.com/gnosia93/eks-agentic-ai/refs/heads/main/code/qwen_distill.py

### Quantization (양자화) ###
* https://raw.githubusercontent.com/gnosia93/eks-agentic-ai/refs/heads/main/code/qwen_quantize.py


> [!Note]
> 모델 사이즈 줄이기
> 
> 모델 사이즈를 실제로 줄이는 방법에는 세가지 기법이 있다. 구조적 pruning은 레이어나 어텐션 헤드 자체를 제거하여 파라미터 수를 줄이는 방식이고, 양자화(Quantization)는 BF16에서 INT8이나 INT4로 데이터 타입을 변환하여 파라미터당 바이트 수를 줄이는 방식이다. 증류(Distillation)는 큰 모델(Teacher)의 지식을 작은 모델(Student)로 전달하여 처음부터 작은 모델을 만드는 방식이다. 이 중 양자화가 가장 간단하고 품질 손실 대비 효과가 크기 때문에 실무에서 가장 널리 사용된다.

## 레퍼런스 ##
* https://github.com/NVIDIA/Model-Optimizer/tree/main 
* https://developer.nvidia.com/blog/optimizing-llms-for-performance-and-accuracy-with-post-training-quantization/
