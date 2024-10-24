import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig

# 1. ClipProjectクラスの定義
class ClipProject(nn.Module):
    def __init__(self):
        super(ClipProject, self).__init__()
        # 最初の線形層: [3840, 512]
        self.layer1 = nn.Linear(512, 3840)
        # 2番目の線形層: [7680, 3840]
        self.layer2 = nn.Linear(3840, 7680)

    def forward(self, x):
        # layer1を通過
        x = self.layer1(x)
        # layer2を通過
        x = self.layer2(x)
        return x

# 2. モデルの初期化
clip_project = ClipProject()

# 3. LoRAの設定
lora_config = LoraConfig(
    r=4,  # LoRAのランク（低ランク次元）
    lora_alpha=16,  # LoRAのスケールファクター
    target_modules=["layer2"],  # LoRAを適用するモジュールを指定
    lora_dropout=0.1  # 過学習を防ぐためのドロップアウト
)

# 4. モデルにLoRAを適用
lora_model = get_peft_model(clip_project, lora_config)

# 5. サンプル入力データ（512次元の特徴量を持つデータ）
input_data = torch.randn(1, 512)

# 6. フォワードパス
output = lora_model(input_data)

# 7. 出力結果の形状を確認
print(f"Output shape: {output.shape}")  # torch.Size([1, 7680])

# 8. モデル内のすべての層の名前、パラメータの形状、および学習可能かどうかを表示
print("\nAll layers, their parameter shapes, and if they are trainable:")
for name, param in lora_model.named_parameters():
    print(f"Layer: {name} | Shape: {param.shape} | Trainable: {param.requires_grad}")
