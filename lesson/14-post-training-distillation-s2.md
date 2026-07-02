학생 모델(1B) 역시 단순히 이미지뿐만 아니라 비디오(시간축), 보이스(주파수축)를 모두 받아먹을 수 있도록 커넥터를 다각화해야 합니다. 각각의 전용 인코더를 1B 언어 모델 백본에 정렬시키는 확장 아키텍처 구조입니다.

```
[Vision Encoder (SigLIP)] ➡️ [MLP Projector 1] ──┐
[Audio Encoder (Whisper)]  ➡️ [MLP Projector 2] ──┼─➡️ [Qwen/Llama 1B Backbone]
[Video Encoder (ViViT)]    ➡️ [MLP Projector 3] 
```

```
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

class OmniProjector(nn.Module):
    """모달리티별 고유 차원을 LLM 공간(1536)으로 튜닝하는 공용 커넥터"""
    def __init__(self, input_dim, llm_dim=1536):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )
    def forward(self, x): return self.net(x)

class StudentOmniVLM1B(nn.Module):
    def __init__(self, llm_id="Qwen/Qwen2.5-1.5B-Instruct"):
        super().__init__()
        # 1B 경량 백본 언어 모델
        self.llm = AutoModelForCausalLM.from_pretrained(llm_id)
        llm_dim = self.llm.config.hidden_size # 1536
        
        # 각 모달리티별 커넥터 배치
        self.image_projector = OmniProjector(input_dim=1152, llm_dim=llm_dim) # SigLIP 차원 대응
        self.audio_projector = OmniProjector(input_dim=1280, llm_dim=llm_dim) # Whisper-large 차원 대응
        self.video_projector = OmniProjector(input_dim=768, llm_dim=llm_dim)  # 비디오 인코더 차원 대응

        # 인코더들은 가중치를 Freeze하고 프로젝트 레이어만 학습 가능하게 세팅
        # (실제 가동 시 외부 인코더 인스턴스를 서브 모듈로 포용합니다.)

    def forward(self, image_embeds, audio_embeds, video_embeds, input_ids, labels=None):
        # 각각의 커넥터를 통과시켜 LLM 공간으로 통일
        img_tokens = self.image_projector(image_embeds) if image_embeds is not None else None
        aud_tokens = self.audio_projector(audio_embeds) if audio_embeds is not None else None
        vid_tokens = self.video_projector(video_embeds) if video_embeds is not None else None
        
        # 텍스트 토큰 임베딩 변환
        text_tokens = self.llm.get_input_embeddings()(input_ids)
        
        # 모든 모달리티 토큰 텐서를 시퀀스 차원(dim=1)으로 결합(Concat)
        token_list = [t for t in [img_tokens, aud_tokens, vid_tokens, text_tokens] if t is not None]
        fused_embeddings = torch.cat(token_list, dim=1)
        
        # LLM 포워드 가동
        return self.llm(inputs_embeds=fused_embeddings, labels=labels)
```
