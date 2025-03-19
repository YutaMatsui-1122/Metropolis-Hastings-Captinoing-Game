from transformers import GPT2LMHeadModel, PreTrainedModel, GPT2Tokenizer
import torch
import torch.nn as nn
from typing import Optional
import os
from typing import Tuple
from huggingface_hub import HfApi, HfFolder

class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

class HuggingFaceClipCaptionModel(PreTrainedModel):
    config_class = GPT2LMHeadModel.config_class

    def __init__(self, config, prefix_length: int, clip_length: Optional[int] = None, 
                 prefix_size: int = 512, num_layers: int = 8, mapping_type: str = 'mlp'):
        super().__init__(config)
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel(config)
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        self.mapping_type = mapping_type

        if mapping_type == 'mlp':
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2,
                                     self.gpt_embedding_size * prefix_length))

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    @classmethod
    def from_pretrained_weights(cls, state_dict, prefix_length: int, mapping_type: str = 'mlp', config_name: str = 'gpt2'):
        config = GPT2LMHeadModel.from_pretrained(config_name).config
        model = cls(config, prefix_length, mapping_type=mapping_type)
        model.load_state_dict(state_dict, strict=False)
        return model
    
# 学習済みモデルとトークナイザをHugging Face Hubにアップロードする関数
def push_to_hub(model, tokenizer, repo_id: str):
    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)

if __name__ == "__main__":
    pretrained_model_dirs = ["models/coco_2017_common_person_only/COCO_A/clipcap", "models/coco_2017_common_person_only/COCO_B/clipcap"]
    
    for name, pretrained_model_dir in zip(["a", "b"], pretrained_model_dirs):
        pretrained_model_path = os.path.join(pretrained_model_dir, "clipcap_019.pt")

        pretrained_model_state_dict = torch.load(pretrained_model_path)

        # モデルをロード
        model = HuggingFaceClipCaptionModel.from_pretrained_weights(
            pretrained_model_state_dict, prefix_length=10, mapping_type='mlp'
        )

        for n, param in model.named_parameters():
            if len(param.shape) == 1:
                print(n, param.shape, param[0])
            else:
                print(n, param.shape, param[0][0])

        # トークナイザの準備
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        # Hugging Face Hubにアップロード
        # repo_id = "ymatsui1122/clipcap_person_only_coco_b"
        repo_id = f"ymatsui1122/clipcap_person_only_coco_{name}"
        push_to_hub(model, tokenizer, repo_id)

        print(f"モデルとトークナイザを{repo_id}にアップロードしました！")