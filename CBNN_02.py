import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot

class BioEncoder(nn.Module):
    """生物信号编码模块"""
    def __init__(self):
        super().__init__()
        # 本体感觉编码分支 (38%)
        self.proprio_fc = nn.Sequential(
            nn.Linear(300, 1140),  # 38% of 3000
            nn.Dropout(0.16),
            nn.ReLU()
        )
        # 运动指令编码分支 (24%)
        self.motor_tcn = nn.Sequential(
            nn.Conv1d(1, 72, kernel_size=3, padding=1),  # 24% of 300
            nn.AdaptiveMaxPool1d(100),
            nn.ReLU()
        )
        
    def forward(self, proprio, motor):
        proprio_out = self.proprio_fc(proprio)
        motor_out = self.motor_tcn(motor.unsqueeze(1)).view(motor.size(0), -1)
        return torch.cat([proprio_out, motor_out], dim=1)

class DynamicPurkinje(nn.Module):
    """动态浦肯野细胞层"""
    def __init__(self):
        super().__init__()
        self.gru = nn.GRUCell(1860, 100)  # 1140+720
        self.inhibit = nn.Linear(100, 100, bias=False)
        
    def forward(self, x, h_state=None):
        if h_state is None:
            h = torch.zeros(x.size(0), 100).to(x.device)
        else:
            h = self.gru(x, h_state)
        return F.relu(h - 0.16*self.inhibit(h))  # 16%抑制

class CerebellarTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入编码层
        self.bio_encoder = BioEncoder()
        
        # 颗粒细胞层 (N=3000)
        self.granule_proj = nn.Linear(1860, 3000)
        self.dropout = nn.Dropout(0.16)
        
        # 浦肯野层 (M=100)
        self.purkinje = DynamicPurkinje()
        
        # 深部小脑核融合
        self.nuclei_fusion = nn.Parameter(torch.tensor([0.6, 0.4]))  # 6%直接+16%间接
        
        # Transformer集成
        self.transformer = nn.TransformerEncoderLayer(
            d_model=256, nhead=4, dim_feedforward=1024
        )
        self.output_layer = nn.Linear(256, 3)  # J1/Motor/Post三输出
        
    def forward(self, proprio_input, motor_input):
        # 生物信号编码
        bio_feat = self.bio_encoder(proprio_input, motor_input)
        
        # 颗粒层扩展
        granule_out = F.relu(self.granule_proj(bio_feat))
        granule_out = self.dropout(granule_out)
        
        # 浦肯野动态处理
        purkinje_out = self.purkinje(granule_out)
        
        # 深部核融合 (公式见注释)
        nuclei_out = (self.nuclei_fusion[0] * granule_out[:, :180] +  # 6%直接通路
                     self.nuclei_fusion[1] * purkinje_out)            # 16%间接通路
        
        # Transformer处理
        transformer_out = self.transformer(nuclei_out.unsqueeze(0))
        
        return self.output_layer(transformer_out.squeeze(0))

# 结构可视化
model = CerebellarTransformer()
x1 = torch.randn(1, 300)  # 本体感觉输入
x2 = torch.randn(1, 100)  # 运动指令输入
dot = make_dot(model(x1, x2), params=dict(model.named_parameters()))
dot.render('cerebellar_net', format='png')  # 生成结构图