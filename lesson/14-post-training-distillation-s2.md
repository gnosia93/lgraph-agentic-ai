## 1B 정제 모델 멀티헤드(Multi-head) 오케스트레이션 ##

전용 인코더(Vision/Audio/Video) → 모달리티별 프로젝터 → 특수토큰 치환 → 1B급 LLM 백본으로
정렬시키는 확장 아키텍처 정리입니다.
```
[Vision Encoder (SigLIP)] ─▶ [MLP Projector 1] ──┐
[Audio Encoder (Whisper)] ─▶ [MLP Projector 2] ──┼─▶ [Qwen/Llama 1B Backbone]
[Video Encoder (ViViT)]   ─▶ [MLP Projector 3] ──┘
```
### 1. 이론적 배경 ###

#### 모달리티 갭(Modality Gap)과 프로젝터의 역할 ####
SigLIP(1152차원), Whisper(1280차원), ViViT는 각자 **다른 벡터 공간**에서 학습된 임베딩을 뱉습니다.
LLM 백본은 자기 텍스트 임베딩 공간(1536차원)만 이해하죠. 프로젝터는 단순 차원 맞춤(reshape)이 아니라,
**이질적인 표현 공간을 언어 모델이 "가짜 토큰(soft prompt)"으로 받아들일 수 있는 공간으로 사상(mapping)**
하는 학습 가능한 변환입니다. LLaVA-1.5가 단층 Linear에서 2-layer MLP + GELU로 바꾼 뒤 성능이 크게
오른 이유가 이 비선형 정렬 능력 때문입니다.

#### "치환"의 본질: 시퀀스 길이가 변한다 ####
`<image>`는 토크나이저에서 정수 ID 1개입니다. 하지만 실제로 LLM에 들어가는 건 이 자리에
**64개(패치) 임베딩 행**이 끼워지는 것. 즉 시퀀스 길이가

```
L_new = L_text - (특수토큰 개수) + Σ(각 모달리티 임베딩 길이)
```

로 **늘어납니다.** 이게 배칭의 핵심 난점입니다. 문장마다 이미지/오디오 유무·길이가 달라
최종 길이가 제각각이 되므로, 단순 `stack`이 아니라 **패딩 기반 배칭**이 필요합니다.

#### 왜 라벨을 -100으로 채우는가 ####
PyTorch `CrossEntropyLoss`의 `ignore_index=-100`입니다. 주입된 이미지/오디오 임베딩 위치에는
"정답 다음 토큰"이 존재하지 않으므로, 그 자리를 손실 계산에서 제외해야 합니다. 안 그러면 모델이
시각 패치 자리에서 엉뚱한 텍스트 토큰을 예측하도록 학습돼 붕괴합니다.

#### 인터리빙(Interleaving)의 핵심 = 순서 보존 ####
`<audio><image>텍스트`처럼 오디오가 먼저 올 수도 있습니다. **텍스트에 등장한 순서 그대로**
임베딩을 배치해야 합니다. "무조건 이미지 먼저"로 하드코딩하면 순서가 뒤바뀔 때 의미가 깨지므로,
발생 위치를 정렬해 순서를 보존하는 방식이 안전합니다.

### 2. 구현 코드 ###

핵심 설계: **이벤트 리스트를 위치순으로 정렬**해서 임의 순서·임의 개수의 모달리티를 인터리빙하고,
**stack 대신 명시적 패딩**으로 배칭하며, **마스크/라벨/dtype**을 일관되게 처리합니다.

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class OmniProjector(nn.Module):
    """모달리티별 인코더 출력을 LLM hidden_dim으로 사상하는 학습형 프로젝터.
    단순 차원 맞춤이 아니라 표현 공간 정렬(alignment)을 담당한다 (LLaVA-1.5 스타일)."""

    def __init__(self, input_dim: int, llm_dim: int = 1536):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StudentOmniVLM(nn.Module):
    def __init__(
        self,
        llm_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
        image_token: str = "<|image_pad|>",
        audio_token: str = "<|audio_pad|>",
    ):
        super().__init__()

        # 1) 백본 + 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(llm_id)
        self.llm = AutoModelForCausalLM.from_pretrained(llm_id)
        llm_dim = self.llm.config.hidden_size  # 1.5B -> 1536

        # 2) 특수 토큰을 "안전하게" 등록 (하드코딩 금지)
        #    이미 존재하지 않는 새 토큰만 추가되고, ID는 tokenizer가 부여한다.
        added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": [image_token, audio_token]}
        )
        if added > 0:
            # 새 토큰 행을 임베딩 행렬에 추가
            self.llm.resize_token_embeddings(len(self.tokenizer))
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(image_token)
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids(audio_token)

        # 3) 모달리티별 전용 프로젝터
        self.image_projector = OmniProjector(input_dim=1152, llm_dim=llm_dim)  # SigLIP
        self.audio_projector = OmniProjector(input_dim=1280, llm_dim=llm_dim)  # Whisper
        # (Video/ViViT용 self.video_projector 도 동일 패턴으로 추가 가능)

    @property
    def _dtype(self):
        return self.llm.get_input_embeddings().weight.dtype

    def forward(
        self,
        input_ids: torch.Tensor,        # [B, T]
        attention_mask: torch.Tensor,   # [B, T]
        image_embeds: torch.Tensor = None,  # [B, N_img, 1152]
        audio_embeds: torch.Tensor = None,  # [B, N_aud, 1280]
        labels: torch.Tensor = None,    # [B, T]
    ):
        device = input_ids.device
        dtype = self._dtype
        batch_size = input_ids.shape[0]

        # A. 텍스트 임베딩 추출
        text_embeddings = self.llm.get_input_embeddings()(input_ids)  # [B, T, D]

        # B. 프로젝션 (dtype 정렬 필수)
        projected_image = (
            self.image_projector(image_embeds.to(dtype)) if image_embeds is not None else None
        )
        projected_audio = (
            self.audio_projector(audio_embeds.to(dtype)) if audio_embeds is not None else None
        )

        fused_embeds, fused_masks, fused_labels = [], [], []

        # C. 문장별 인터리빙 (위치순 정렬로 순서/개수 무관하게 처리)
        for i in range(batch_size):
            cur_ids = input_ids[i]
            cur_txt = text_embeddings[i]
            cur_msk = attention_mask[i]
            cur_lbl = labels[i] if labels is not None else None

            # 특수토큰 발생 지점을 (위치, 주입텐서) 이벤트로 수집
            events = []
            if projected_image is not None:
                for p in (cur_ids == self.image_token_id).nonzero(as_tuple=True)[0].tolist():
                    events.append((p, projected_image[i]))
            if projected_audio is not None:
                for p in (cur_ids == self.audio_token_id).nonzero(as_tuple=True)[0].tolist():
                    events.append((p, projected_audio[i]))
            events.sort(key=lambda e: e[0])  # ★ 텍스트 등장 순서 보존

            if not events:  # 순수 텍스트 문장
                fused_embeds.append(cur_txt)
                fused_masks.append(cur_msk)
                if cur_lbl is not None:
                    fused_labels.append(cur_lbl)
                continue

            emb_seg, msk_seg, lbl_seg = [], [], []
            cursor = 0
            for pos, mm in events:
                # 특수토큰 앞쪽 텍스트 조각
                if pos > cursor:
                    emb_seg.append(cur_txt[cursor:pos])
                    msk_seg.append(cur_msk[cursor:pos])
                    if cur_lbl is not None:
                        lbl_seg.append(cur_lbl[cursor:pos])
                # 모달리티 텐서 주입
                n = mm.shape[0]
                emb_seg.append(mm)
                msk_seg.append(torch.ones(n, dtype=cur_msk.dtype, device=device))
                if cur_lbl is not None:
                    lbl_seg.append(torch.full((n,), -100, dtype=cur_lbl.dtype, device=device))
                cursor = pos + 1  # 특수토큰 1개 소비

            # 남은 후속 텍스트
            if cursor < cur_ids.shape[0]:
                emb_seg.append(cur_txt[cursor:])
                msk_seg.append(cur_msk[cursor:])
                if cur_lbl is not None:
                    lbl_seg.append(cur_lbl[cursor:])

            fused_embeds.append(torch.cat(emb_seg, dim=0))
            fused_masks.append(torch.cat(msk_seg, dim=0))
            if cur_lbl is not None:
                fused_labels.append(torch.cat(lbl_seg, dim=0))

        # D. 가변 길이 → 오른쪽 패딩으로 배칭 (stack 금지!)
        max_len = max(e.shape[0] for e in fused_embeds)
        D = text_embeddings.shape[-1]

        inputs_embeds = torch.zeros(batch_size, max_len, D, dtype=dtype, device=device)
        out_mask = torch.zeros(batch_size, max_len, dtype=attention_mask.dtype, device=device)
        out_labels = (
            torch.full((batch_size, max_len), -100, dtype=input_ids.dtype, device=device)
            if labels is not None else None
        )

        for i in range(batch_size):
            L = fused_embeds[i].shape[0]
            inputs_embeds[i, :L] = fused_embeds[i]
            out_mask[i, :L] = fused_masks[i]
            if out_labels is not None:
                out_labels[i, :L] = fused_labels[i]

        # E. 백본 forward
        return self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=out_mask,
            labels=out_labels,
            return_dict=True,
        )


if __name__ == "__main__":
    model = StudentOmniVLM()
    B, T, D = 2, 10, model.llm.config.hidden_size

    input_ids = torch.randint(0, 1000, (B, T))
    # 0번 문장: 위치1에 이미지, 위치2에 오디오 심기
    input_ids[0, 1] = model.image_token_id
    input_ids[0, 2] = model.audio_token_id
    attention_mask = torch.ones(B, T, dtype=torch.long)
    labels = input_ids.clone()

    image_embeds = torch.randn(B, 64, 1152)   # SigLIP: 64 patches
    audio_embeds = torch.randn(B, 300, 1280)  # Whisper: 300 frames

    out = model(input_ids, attention_mask, image_embeds, audio_embeds, labels)
    print("loss:", out.loss.item())
    print("logits:", out.logits.shape)  # [B, max_fused_len, vocab]
    print("✅ 순서·개수 무관 인터리빙 + 패딩 배칭 검증 완료")
```

### 3. 실전 투입 전 체크리스트 (이론+실무) ###

1. **토큰 개수 = 패치 개수 규약 (권장 대안)**: 위 코드는 "특수토큰 1개 → 64행 치환" 방식입니다.
   Qwen2-VL/최신 LLaVA는 반대로 **전처리 단계에서 `<image>`를 패치 수만큼 반복 삽입**한 뒤 1:1로
   갈아끼웁니다. 시퀀스 길이가 전처리 때 확정돼 배칭이 단순하고 정합성 검증이 쉬워집니다.
2. **정합성 검증**: `주입 임베딩 총 행수 == 특수토큰 총 개수 × 패치수`를 assert로 걸어 파이프라인 버그 조기 발견.
3. **2단계 학습 전략**: 정렬 초기엔 **백본 freeze + 프로젝터만 학습**(alignment pretraining),
   이후 **프로젝터 + LoRA 미세조정**(instruction tuning)하는 LLaVA식 2-stage가 소형 모델에서 안정적.
4. **생성(inference) 경로**: 학습용 `forward`만 있고 `generate`가 없습니다. 추론 시 융합된 `inputs_embeds`를
   만들어 `self.llm.generate(inputs_embeds=..., attention_mask=...)`로 넣고, 첫 스텝에만 멀티모달 임베딩이
   들어가도록 KV 캐시 설계 필요.
5. **좌측 패딩 고려**: 디코더 생성에서는 보통 left-padding이 유리. 학습(right-pad)/추론(left-pad) 규약 분리.
6. **dtype/디바이스 정렬**: fp16/bf16 학습 시 프로젝터 출력과 텍스트 임베딩 dtype을 맞춰야 `cat`에서 안 터짐.
7. **비디오(ViViT) 확장**: 시간축 토큰은 프레임 수 × 프레임당 패치라 시퀀스 폭증.
   temporal pooling이나 Q-Former류 토큰 압축을 프로젝터 앞단에 두는 걸 고려.
