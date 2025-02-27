import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# ------------------ 1. 神经网络定义 ------------------
class ObservationActionTransformer(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        
        # MLP处理分支
        self.mlp = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 300),
            nn.ReLU(),
            nn.Linear(300, 20)
        )
        
        # 输入投影层
        self.obs_proj = nn.Linear(obs_dim, 20)
        self.act_proj = nn.Linear(act_dim, 20)
        
        # Transformer组件
        self.pos_embedding = nn.Parameter(torch.randn(3, 1, 20))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=20,
            nhead=4,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        # 输出层
        self.final_fc = nn.Linear(3*20, 1)

    def forward(self, obs, act):
        batch_size = obs.size(0)
        
        # 处理各路径
        mlp_out = self.mlp(torch.cat([obs, act], dim=1))
        proj_obs = self.obs_proj(obs)
        proj_act = self.act_proj(act)
        
        # 构建序列并添加位置编码
        seq = torch.stack([mlp_out, proj_obs, proj_act], dim=0) + self.pos_embedding
        
        # Transformer处理
        trans_out = self.transformer(seq)
        
        # 输出处理
        return self.final_fc(trans_out.permute(1,0,2).reshape(batch_size, -1))

# ------------------ 2. 数据集类定义 ------------------
class OATDataset(Dataset):
    def __init__(self, num_samples, obs_dim=10, act_dim=5):
        # 示例数据生成（实际使用时替换为真实数据）
        self.obs = torch.randn(num_samples, obs_dim)
        self.act = torch.randn(num_samples, act_dim)
        # 示例目标：线性组合 + 噪声
        self.targets = (
            0.7 * self.obs[:, 0] 
            + 1.5 * self.act[:, 1] 
            + 0.1 * torch.randn(num_samples)
        ).unsqueeze(1)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return {
            'obs': self.obs[idx],
            'act': self.act[idx],
            'target': self.targets[idx]
        }

# ------------------ 3. 训练函数 ------------------
def train(model, device, train_loader, val_loader, epochs, lr):
    # 初始化训练组件
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
    best_val_loss = float('inf')
    
    # 训练记录
    history = {'train': [], 'val': []}
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            obs = batch['obs'].to(device)
            act = batch['act'].to(device)
            target = batch['target'].to(device)
            
            output = model(obs, act)
            loss = criterion(output, target)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()
            
            train_loss += loss.item() * obs.size(0)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                obs = batch['obs'].to(device)
                act = batch['act'].to(device)
                target = batch['target'].to(device)
                
                output = model(obs, act)
                val_loss += criterion(output, target).item() * obs.size(0)
        
        # 计算平均损失
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        
        # 学习率调整
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        # 打印进度
        print(f'Epoch {epoch+1:03d} | '
              f'Train Loss: {train_loss:.4f} | '
              f'Val Loss: {val_loss:.4f} | '
              f'LR: {optimizer.param_groups[0]["lr"]:.2e}')

    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(history['train'], label='Training Loss')
    plt.plot(history['val'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_curve.png')
    plt.close()
    
    return model

# ------------------ 4. 主程序 ------------------
def main():
    # 超参数配置
    config = {
        'obs_dim': 10,
        'act_dim': 5,
        'batch_size': 64,
        'epochs': 100,
        'lr': 1e-3,
        'num_samples': 5000
    }
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 准备数据
    full_dataset = OATDataset(config['num_samples'], 
                            obs_dim=config['obs_dim'], 
                            act_dim=config['act_dim'])
    train_size = int(0.8 * len(full_dataset))
    train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, len(full_dataset)-train_size])
    
    train_loader = DataLoader(train_set, 
                             batch_size=config['batch_size'], 
                             shuffle=True,
                             num_workers=2)
    val_loader = DataLoader(val_set, 
                           batch_size=config['batch_size'], 
                           shuffle=False)
    
    # 初始化模型
    model = ObservationActionTransformer(config['obs_dim'], config['act_dim'])
    model = model.to(device)
    
    # 开始训练
    trained_model = train(model, device, train_loader, val_loader, 
                         config['epochs'], config['lr'])
    
    # 保存最终模型
    torch.save(trained_model.state_dict(), 'final_model.pth')

if __name__ == "__main__":
    main()