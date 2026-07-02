학생 모델(1B) 역시 단순히 이미지뿐만 아니라 비디오(시간축), 보이스(주파수축)를 모두 받아먹을 수 있도록 커넥터를 다각화해야 합니다. 각각의 전용 인코더를 1B 언어 모델 백본에 정렬시키는 확장 아키텍처 구조입니다.

```
[Vision Encoder (SigLIP)] ➡️ [MLP Projector 1] ──┐
[Audio Encoder (Whisper)]  ➡️ [MLP Projector 2] ──┼─➡️ [Qwen/Llama 1B Backbone]
[Video Encoder (ViViT)]    ➡️ [MLP Projector 3] 
```

### 멀티헤드 오케스트레이션의 핵심 메커니즘 ###
우리가 텍스트 입력으로 "<image><audio>이 파일들을 분석해라."라고 주면, 텍스트 토크나이저는 <image>와 <audio>라는 특수 토큰을 단 1개의 정수 ID로 인코딩합니다.
우리가 해야 할 일은 LLM 디코더로 들어가기 전에, 그 1글자짜리 텍스트 임베딩 자리를 찾아내서 비전 인코더가 뽑아낸 수백 개의 시각 패치 텐서 덩어리로 통째로 '치환(Replace)' 및 '슬라이싱(Slicing)' 가공을 해주어야 합니다.

실제 오픈소스 진영(LLaVA, Qwen2-Audio 등)에서 다중 모달리티를 인젝션할 때 가장 골치 아픈 점은 **“어느 자리에 이미지 토큰을 넣고, 어느 자리에 텍스트와 오디오 토큰을 치환해 넣을 것인가? 이다.

### 💻 1B 소형 정제 모델을 위한 완전판 오케스트레이션 코드 ###
```
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class OmniProjector(nn.Module):
    """모달리티별 텐서 차원을 LLM 숨겨진 차원(1536)으로 프로젝션"""
    def __init__(self, input_dim, llm_dim=1536):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )
    def forward(self, x): 
        return self.net(x)

class StudentOmniVLM1B(nn.Module):
    def __init__(self, llm_id="Qwen/Qwen2.5-1.5B-Instruct", image_token_id=151646, audio_token_id=151647):
        super().__init__()
        # 1. 1B 경량 백본 언어 모델 로드
        self.llm = AutoModelForCausalLM.from_pretrained(llm_id)
        llm_dim = self.llm.config.hidden_size # 1536
        
        # 2. 토크나이저 내의 특수 토큰 ID 매핑 점검 (치환 포인트 포착용)
        self.image_token_id = image_token_id
        self.audio_token_id = audio_token_id
        
        # 3. 모달리티별 멀티헤드 전용 프로젝트 레이어
        self.image_projector = OmniProjector(input_dim=1152, llm_dim=llm_dim) # SigLIP 대응
        self.audio_projector = OmniProjector(input_dim=1280, llm_dim=llm_dim) # Whisper 대응

    def forward(self, image_embeds, audio_embeds, input_ids, attention_mask, labels=None):
        """
        [Shape 정렬 목표]
        - input_ids: [Batch, Text_Seq_Len]
        - image_embeds: [Batch, Image_Patch_Len(예: 64), 1152]
        - audio_embeds: [Batch, Audio_Frame_Len(예: 300), 1280]
        """
        batch_size, text_seq_len = input_ids.shape
        device = input_ids.device
        
        # A. 원본 텍스트 임베딩 추출 -> [Batch, Text_Seq_Len, 1536]
        text_embeddings = self.llm.get_input_embeddings()(input_ids)
        
        # B. 외부 가속기 토큰들을 LLM 차원으로 미세 조율
        # [Batch, 64, 1536]
        projected_image = self.image_projector(image_embeds) if image_embeds is not None else None
        # [Batch, 300, 1536]
        projected_audio = self.audio_projector(audio_embeds) if audio_embeds is not None else None

        # C. [핵심 패치] 배치의 각 문장별로 순회하며 특수 토큰 자리에 멀티모달 텐서 융합 인터리빙(Interleaving)
        final_embeddings = []
        final_labels = [] if labels is not None else None
        
        for i in range(batch_size):
            cur_input_ids = input_ids[i]
            cur_text_embeds = text_embeddings[i]
            
            # 1단계: 문장 내에서 이미지 특수 토큰과 오디오 특수 토큰 위치(인덱스) 스캔
            img_loc = (cur_input_ids == self.image_token_id).nonzero(as_tuple=True)[0]
            aud_loc = (cur_input_ids == self.audio_token_id).nonzero(as_tuple=True)[0]
            
            # 실전 인프라 예외처리: 특수 토큰이 없는 일반 텍스트 입력 문장일 경우
            if len(img_loc) == 0 and len(aud_loc) == 0:
                final_embeddings.append(cur_text_embeds)
                if labels is not None:
                    final_labels.append(labels[i])
                continue
                
            # 2단계: 조각난 텍스트와 가속기 멀티헤드 텐서들을 순서대로 조립
            # 시퀀스를 분할 조립하기 위한 세그먼트 빌더 구현
            segments = []
            cur_idx = 0
            
            # 본 코드에서는 단순화를 위해 하나의 문장에 이미지 1개, 오디오 1개가 맨 앞에 순서대로 온다고 가정한 슬라이싱 로직 매핑
            # <image> 토큰 이전 텍스트 분할 추출 (보통 0번째 위치에 특수 토큰이 위치함)
            if projected_image is not None and len(img_loc) > 0:
                idx = img_loc[0].item()
                if idx > cur_idx: segments.append(cur_text_embeds[cur_idx:idx])
                segments.append(projected_image[i]) # 치환: 1글자 텍스트 임베딩 대신 64줄의 이미지 텐서 주입
                cur_idx = idx + 1
                
            if projected_audio is not None and len(aud_loc) > 0:
                idx = aud_loc[0].item()
                if idx > cur_idx: segments.append(cur_text_embeds[cur_idx:idx])
                segments.append(projected_audio[i]) # 치환: 1글자 텍스트 임베딩 대신 300줄의 오디오 텐서 주입
                cur_idx = idx + 1
                
            # 남은 후속 텍스트 전부 결합
            if cur_idx < text_seq_len:
                segments.append(cur_text_embeds[cur_idx:])
                
            # 문장별 조립 완료 -> [New_Fused_Seq_Len, 1536]
            fused_sentence_embed = torch.cat(segments, dim=0)
            final_embeddings.append(fused_sentence_embed)
            
            # 3단계: Label(정답) 생성 시에도 가속기 임베딩이 침범한 만큼 패딩값(-100)으로 축 확장
            if labels is not None:
                cur_labels = labels[i]
                label_segments = []
                cur_lbl_idx = 0
                
                if len(img_loc) > 0:
                    idx = img_loc[0].item()
                    if idx > cur_lbl_idx: label_segments.append(cur_labels[cur_lbl_idx:idx])
                    # 이미지 토큰 영역은 손실(Loss) 계산에서 제외하도록 익스클루전 마스크(-100) 채우기
                    label_segments.append(torch.full((projected_image.shape[1],), -100, dtype=cur_labels.dtype, device=device))
                    cur_lbl_idx = idx + 1
                    
                if len(aud_loc) > 0:
                    idx = aud_loc[0].item()
                    if idx > cur_lbl_idx: label_segments.append(cur_labels[cur_lbl_idx:idx])
                    # 오디오 영역 제외 마스크 채우기
                    label_segments.append(torch.full((projected_audio.shape[1],), -100, dtype=cur_labels.dtype, device=device))
                    cur_lbl_idx = idx + 1
                    
                if cur_lbl_idx < text_seq_len:
                    label_segments.append(cur_labels[cur_lbl_idx:])
                    
                final_labels.append(torch.cat(label_segments, dim=0))

        # D. 가변 길이 문장들의 최종 배칭 처리를 위한 스택(Stack) 파이프라이닝
        # (실제 대량 분산 배치 전파 시에는 크기를 통일하는 패딩 작업이 수행되어야 합니다.)
        inputs_embeds = torch.stack(final_embeddings, dim=0)
        
        final_attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=attention_mask.dtype, device=device)
        
        train_labels = None
        if labels is not None:
            train_labels = torch.stack(final_labels, dim=0)

        # E. 드디어 완벽한 행렬 형태로 정렬된 텐서들을 Qwen/Llama 백본에 바인딩
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=final_attention_mask,
            labels=train_labels,
            return_dict=True
        )
        return outputs

if __name__ == "__main__":
    # 아키텍처 단위 검증용 더미 가동 코드
    model = StudentOmniVLM1B()
    print("✅ 특수 토큰 동적 치환 엔진(Interleaving)을 포함한 완전판 멀티헤드 VLM 조립 완료.")
```
