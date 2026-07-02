구조 설계의 핵심은 **"어떻게 영리하게 다른 차원의 가속기 텐서들을 연결할 것인가"**입니다.
구글의 최신 비전 인코더인 SigLIP(추출 벡터 차원: 1152)의 시각 특징을 경량 백본인 Qwen2.5-1.5B-Instruct(숨겨진 차원: 1536) 내부에 매핑하는 LLaVA 스타일의 MLP 커넥터(프로젝션 레이어) 구현 소스 코드입니다.

```
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, AutoModelForCausalLM

class VLM1BConnector(nn.Module):
    """
    SigLIP (Vision Encoder)와 Qwen/Llama (LLM BackBone)의 차원을 매핑해주는 
    LLaVA 스타일의 고성능 MLP 프로젝션 커넥터 레이어
    """
    def __init__(self, vision_dim: int = 1152, llm_dim: int = 1536):
        super().__init__()
        # 단순 Linear 레이어보다 멀티모달 정렬 레이어에는 중간에 겔루(GELU)가 들어간 
        # 2층 구조의 MLP가 벤치마크 점수를 훨씬 잘 뽑아냅니다.
        self.mlp = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class CustomVLM1BModel(nn.Module):
    def __init__(self, vision_model_id="google/siglip-so400m-patch14-384", llm_model_id="Qwen/Qwen2.5-1.5B-Instruct"):
        super().__init__()
        print(f"👁️ 비전 인코더 로딩 중: {vision_model_id}")
        self.vision_encoder = AutoModel.from_pretrained(vision_model_id).vision_model
        
        print(f"✍️ 경량 언어 모델 백본 로딩 중: {llm_model_id}")
        self.llm = AutoModelForCausalLM.from_pretrained(llm_model_id)
        
        # 임베딩 차원 추출 및 커넥터 연결
        vision_dim = self.vision_encoder.config.hidden_size      # 1152
        llm_dim = self.llm.config.hidden_size                  # 1536
        self.connector = VLM1BConnector(vision_dim=vision_dim, llm_dim=llm_dim)
        
        # [중요 - 페이즈 1 정렬 전략]
        # 비전 인코더와 대형 LLM 가중치는 잠그고(Freeze), 커넥터 레이어만 훈련 모드로 개방
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.llm.parameters():
            param.requires_grad = False
        for param in self.connector.parameters():
            param.requires_grad = True

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        # 1. 시각 특징 추출 (Batch, Sequence_Length, Vision_Dim)
        # SigLIP의 패치 토큰 아웃풋 획득
        vision_outputs = self.vision_encoder(pixel_values).last_hidden_state
        
        # 2. 커넥터를 통해 LLM의 가중치 임베딩 공간 차원으로 매핑 (1152 -> 1536)
        image_features = self.connector(vision_outputs)
        
        # 3. 텍스트를 고유 임베딩 벡터로 변환
        text_embeddings = self.llm.get_input_embeddings()(input_ids)
        
        # 4. 멀티모달 입력 병합: 이미지 토큰 임베딩 레이어 뒤에 텍스트 임베딩 컨캣(Concat)
        # 실제 구현에서는 <image> 토큰이 위치한 인덱스에 정밀하게 삽입해야 하나, 
        # 프리펜드(Prepend) 방식의 단순화 구조로 예시 구현
        inputs_embeds = torch.cat([image_features, text_embeddings], dim=1)
        
        # attention_mask도 이미지 토큰 길이만큼 확장 필요
        batch_size, img_seq_len, _ = image_features.shape
        img_attn_mask = torch.ones((batch_size, img_seq_len), device=input_ids.device)
        full_attention_mask = torch.cat([img_attn_mask, attention_mask], dim=1)

        # 5. LLM 디코더 가동
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=labels
        )
        return outputs

# 모델 초기화 테스트
if __name__ == "__main__":
    model = CustomVLM1BModel()
    print("✅ SigLIP + Qwen 1B 경량 멀티모달 가속 아키텍처 조립 완료.")
```
