## 100B → 1B 멀티모달 지식 증류 가이드 ##

100B급 대형 멀티모달 모델(VLM)을 교사(Teacher)로 삼아 1B급 소형 정제 모델(Student)을 만드는
지식 증류(Knowledge Distillation) + 합성 데이터 생성(Synthetic Data Generation) 파이프라인 가이드입니다.
멀티모달은 텍스트보다 정렬(Alignment)이 까다로워 전략적 접근이 필요합니다.

관련 구현 문서:
- [1단계 - 100B Omni-VLM 추론 파이프라인](https://github.com/gnosia93/lgraph-agentic-ai/blob/main/lesson/14-post-training-distillation-s1.md)
- [2단계 - 1B 멀티헤드 오케스트레이션](https://github.com/gnosia93/lgraph-agentic-ai/blob/main/lesson/14-post-training-distillation-s2.md)
- [3단계 - DeepSpeed ZeRO-3 학습](https://github.com/gnosia93/lgraph-agentic-ai/blob/main/lesson/14-post-training-distillation-s3.md)


### 1단계: 교사 모델(100B VLM)을 통한 데이터 정제 및 합성 ###

1B 모델은 파라미터가 작아 지저분한 데이터를 학습하면 성능이 급격히 무너집니다. 100B 모델을
**'데이터 공장 및 정제소'**로 활용합니다.

- **고품질 캡셔닝 / VQA 생성 (시각 지식 증류)**: LAION 등 대규모 이미지셋에 대해 100B VLM으로
  Dense Captioning과 복잡한 추론형 VQA 쌍을 생성합니다. "생각의 흐름(CoT)을 포함해 인과관계를
  단계별로 설명"처럼 교사의 추론 프로세스 자체를 데이터에 녹입니다.
- **품질 필터링**: 텍스트 리워드 모델만으로는 이미지-텍스트 정합성을 못 잡습니다. **CLIP/SigLIP
  score**(이미지-캡션 정합도), **self-consistency**(동일 질문 다회 생성 후 다수결), **교사 재검증**을
  병행하고, CoT는 환각을 증폭할 수 있으므로 독성/환각 스캔 게이트를 반드시 통과시킵니다.

#### 증류 방식에 따라 인프라가 갈린다 (설계 전 반드시 결정) ####
- **응답(텍스트) 증류**: 교사가 만든 정답 텍스트로 학생을 SFT. 관리형 **API 배치 추론으로 충분**.
- **로짓(soft label) 증류**: 교사의 토큰별 확률분포를 KL로 모방. 대부분의 관리형 API는 전체 vocab
  분포를 주지 않으므로, **교사를 self-host(vLLM/Transformers)** 해서 full logits(또는 top-k logprobs)를
  직접 추출·저장해야 합니다. 저장량이 크므로 보통 **top-k(k=50~100) logprobs**만 저장해 근사합니다.

### 2단계: 소형 모델(1B) 아키텍처 및 정렬(Alignment) 설계 ####

최근 트렌드인 LLaVA 스타일 커넥터 구조를 씁니다. 검증된 "눈"과 "입"을 가져와 그 사이의 얇은 연결부만
학습하는 방식이라, 1B급도 소수의 GPU로 며칠 만에 멀티모달로 진화시킬 수 있습니다.

- **비전/오디오 인코더**: 처음부터 학습하지 않고 검증된 오픈소스 인코더(SigLIP-SO400M(1152차원),
  CLIP-ViT-L, Whisper 등)를 **동결(Freeze)** 해 사용.
- **프로젝션 레이어**: 인코더 출력 벡터를 LLM hidden 차원으로 사상하는 경량 MLP(+GELU).
  단순 차원 변환이 아니라 이질적 표현공간을 정렬하는 핵심 학습 대상.
- **백본(LLM)**: **Llama-3.2-1B**나 **Qwen2.5-1.5B** 같은 검증된 경량 백본에서 시작.

> 로짓 증류를 계획한다면 **교사·학생의 토크나이저 계열을 통일**해야 합니다(예: 둘 다 Qwen 또는 둘 다
> Llama). vocab이 다르면 토큰별 확률분포를 정렬할 수 없어 로짓 KD가 성립하지 않으며, 이 경우
> 응답(시퀀스) 증류로 전환해야 합니다.

### 3단계: 인프라 기반 분산 학습 전략 ###

1B 모델이라도 멀티모달 텐서(고해상도 이미지/비디오/오디오)는 메모리를 크게 잡아먹으므로 학습을
2단계로 분기합니다.

#### ① 정렬 사전학습 (Freeze) ####
- 목표: 이미지/오디오 ↔ 텍스트 벡터 공간 정렬.
- 방식: **비전·오디오 인코더와 LLM을 모두 동결**하고 **프로젝터만** 학습(LLaVA 1단계 정석).
- 인프라: 학습 대상이 작으므로 PyTorch **FSDP**로 싱글 노드 멀티 GPU(L40S/A100)에서 빠르게 완료.

#### ② 인스트럭션 튜닝 / 증류 (Full or LoRA) ####
- 목표: 100B 합성 QA 데이터로 프로젝터 + (전체 또는 LoRA) 백본을 학습.
- 로짓 증류: 교사의 토큰별 확률분포를 KL-Divergence로 모방하면 단순 SFT보다 성능 곡선이 가팔라집니다.
- 인프라: 큰 배치/텐서 처리를 위해 DeepSpeed ZeRO-2/3 + EFA로 노드 간 통신 병목 제거.
  단, **1B 학생 학습은 FSDP나 ZeRO-2로 충분한 경우가 많습니다.** ZeRO-3+오프로드는 통신·전송
  오버헤드가 커서 교사 학습이나 초장문 컨텍스트에서 진가를 발휘합니다.


### 지식 증류(KD) 이론 심화 ###

#### 왜 Soft Label이 강한가 ###
정답 하나만 배우는 hard label과 달리, 교사의 전체 분포(soft label)는 "정답은 A지만 B도 그럴듯,
C는 아님" 같은 **클래스 간 유사도 구조(dark knowledge)** 를 담아 학생에게 훨씬 풍부한 신호를 줍니다.

#### Temperature와 KL 손실 ####
```
L = α · CE(student, hard_label)
  + (1 - α) · T² · KL( softmax(teacher_logits / T) ‖ softmax(student_logits / T) )
```
- **T(temperature)** 를 높이면 분포가 부드러워져 dark knowledge가 더 드러남(보통 T=1~4).
- soft 항 그래디언트가 `1/T²`로 줄어드는 걸 보정하려 `T²`를 곱합니다(Hinton et al.).
- α로 hard/soft 비중을 조절.
- 멀티모달에서는 **텍스트 토큰 위치의 언어 로짓**에만 KD를 적용하고, 주입된 이미지/오디오 임베딩
  위치는 타깃이 없으므로 `-100`으로 마스킹해 제외합니다.


### GPU 메모리 — 원인 분해와 대응 ###

멀티모달 학습에서 VRAM이 폭발하는 이유는 데이터 사이즈와 연산 스케일이 텍스트와 차원이 다르기
때문입니다. 원인별로 분리하면 대응이 명확해집니다.

| 구분 | 원인 | 대응 |
|------|------|------|
| 파라미터/옵티마이저 | 모델·Adam 상태 | ZeRO-2/3, 오프로드, LoRA |
| **활성화(activation)** | 긴 멀티모달 시퀀스의 forward 중간값 | **gradient checkpointing**, FlashAttention |
| 어텐션 맵 | 나이브 어텐션의 N×N 물질화 | **FlashAttention**(활성화 O(N)로 감소) |
| KV 캐시(추론) | 생성 길이 × 레이어 | paged-attention(vLLM), max_len 관리 |

- **시-공간 텐서 크기**: 384×384 이미지 한 장도 SigLIP을 거치면 `[패치 수, 1152]` 텐서가 됩니다.
  비디오는 이 텐서가 시간축으로 수십~수백 장 쌓이고, 오디오는 주파수 축의 롱 시퀀스가 됩니다.
- **어텐션 비용**: 나이브 어텐션은 N×N 맵을 물질화해 **메모리 O(N²)**. 단 **FlashAttention을 쓰면
  활성화 메모리는 O(N)** 으로 선형이 되고, 제곱으로 남는 것은 **연산량(FLOPs)** 입니다. 따라서 롱
  시퀀스 OOM의 1차 방어선은 **FlashAttention + gradient checkpointing** 입니다.
- **학습 시 활성화 누적**: 역전파를 위해 forward의 거대한 중간 텐서를 모두 보관해야 하므로 메모리가
  빠르게 고갈됩니다. 이 활성화 메모리는 ZeRO로는 줄지 않으므로 checkpointing이 사실상 필수입니다.


### 프로젝트 체크리스트 ###

1. **증류 방식 확정**: 응답(텍스트) 증류(API 배치 OK) vs 로짓(soft label) 증류(교사 self-host +
   top-k logprobs 저장 필요). 방식에 따라 교사 인프라 설계가 달라짐.
2. **교사·학생 토크나이저 계열 통일**: 로짓 KD 시 필수. 못 맞추면 응답 증류로 전환.
3. **FlashAttention + gradient checkpointing 활성화**: 멀티모달 롱 시퀀스 OOM의 1차 방어선.
4. **인코더 freeze + 프로젝터 우선 학습**(1단계) → 백본/LoRA 튜닝(2단계).
5. **멀티모달 품질 필터**: CLIP-score + self-consistency + 교사 재검증. 텍스트 리워드 모델만으론 부족.
6. **하드웨어-경량화 매칭**: FP8은 Hopper(H100급) 네이티브. A100은 INT8/AWQ 등 사용.
7. **사전 임베딩 캐싱(.pt)**: 인코더 forward를 매 스텝 반복하지 말고 구워서 로딩 → I/O·연산 병목 감소.
8. **평가 트래킹**: MM-Vet / MMBench로 "교사 대비 % 성능"을 정량 목표(예: 75%)로 설정·추적.
9. **학생 규모에 맞는 병렬화**: 1B 학습은 FSDP/ZeRO-2로 충분한 경우가 많음. ZeRO-3+오프로드는 교사
   학습·초장문에서 활용.
10. **비용/속도 타협**: API 배치(비동기·할인) vs self-host(FP8/AWQ + vLLM 고스루풋). 단 로짓 KD가
    필요하면 self-host가 사실상 강제.
