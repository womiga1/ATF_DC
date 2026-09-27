import torch
import torch.nn as nn

from mamba_ssm import Mamba

def multi_channel_similarity(a, b, c):
    inner_ab = torch.sum(a * b, dim=-1)
    inner_bc = torch.sum(b * c, dim=-1)
    inner_ca = torch.sum(c * a, dim=-1)
    combined = inner_ab + inner_bc + inner_ca
    return torch.softmax(combined, dim=1)

class TriModalInteractiveModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.public_ratio = 0.33
        
        self.public_num = int(in_channels * self.public_ratio)
        self.public_num = (self.public_num // 3) * 3  
        
        self.bn_a = nn.BatchNorm1d(in_channels)
        self.bn_b = nn.BatchNorm1d(in_channels)
        self.bn_c = nn.BatchNorm1d(in_channels)
        self.bn_public = nn.BatchNorm1d(self.public_num*3)
                 
        self.public_fusion = Mamba(
            d_model=self.public_num*3,
            d_state=self.public_num,
            d_conv=4,
            expand=2,
        )
        # self.public_fusion = nn.Conv1d(
        #     in_channels=self.public_num*3,  
        #     out_channels=self.public_num*3,    
        #     kernel_size=3,
        #     padding=1,
        #     stride=1
        # )

    def forward(self, a, b, c):
        B, C, L = a.shape
        
        # 特征标准化
        a_norm = self.bn_a(a)
        b_norm = self.bn_b(b)
        c_norm = self.bn_c(c)
        
        # 计算联合相似度（维度修正）
        similarity = multi_channel_similarity(a_norm, b_norm, c_norm)
        sorted_idx = torch.argsort(similarity, dim=1, descending=True)
        
        # 提取公共特征（使用动态public_num）
        public_a = a_norm.gather(1, 
            sorted_idx[:, :self.public_num, None].expand(-1, -1, L)
        )
        public_b = b_norm.gather(1, 
            sorted_idx[:, :self.public_num, None].expand(-1, -1, L)
        )
        public_c = c_norm.gather(1, 
            sorted_idx[:, :self.public_num, None].expand(-1, -1, L)
        )
        
        # 公共特征融合
        public_concat = torch.cat([public_a, public_b, public_c], dim=1)
        public_concat = public_concat.transpose(1, 2)
        # 确保数据类型为float32，Mamba要求这种类型
        public_concat = public_concat.float()
        public_fused = self.public_fusion(public_concat)
        public_fused = public_fused.transpose(1, 2)

        
        # 均匀分配（确保可分割）
        split_size = self.public_num
        perm_idx = torch.randperm(self.public_num*3, device=a.device)
        public_a_assign = public_fused[:, perm_idx[:split_size], :]
        public_b_assign = public_fused[:, perm_idx[split_size:2*split_size], :]
        public_c_assign = public_fused[:, perm_idx[2*split_size:], :]
        
        # 处理私有特征
        private_size = C - self.public_num
        private_a = a_norm.gather(1, 
            sorted_idx[:, self.public_num:, None].expand(B, private_size, L)
        ).contiguous()
        private_b = b_norm.gather(1, 
            sorted_idx[:, self.public_num:, None].expand(B, private_size, L)
        ).contiguous()
        private_c = c_norm.gather(1, 
            sorted_idx[:, self.public_num:, None].expand(B, private_size, L)
        ).contiguous()
        
        # 最终输出维度保障
        out_a = torch.cat([private_a, public_a_assign], dim=1)
        out_b = torch.cat([private_b, public_b_assign], dim=1)
        out_c = torch.cat([private_c, public_c_assign], dim=1)
        
        return out_a, out_b, out_c