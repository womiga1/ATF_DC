import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .embed import DataEmbedding
from .block import EncoderBlock
from .RevIN import RevIN
from tkinter import _flatten
from model.modules import FFLayer

class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    # def forward(self, series, prior1, prior2, patch_size, attn_mask=None):
    #     series_list = []
    #     prior1_list = []
    #     prior2_list = []
    #     for attn_layer in self.attn_layers:
    #         series, prior1, prior2 = attn_layer(series, prior1, prior2, patch_size, mask=attn_mask)
    #         series_list.append(series.unsqueeze(1))
    #         prior1_list.append(prior1.unsqueeze(1))
    #         prior2_list.append(prior2.unsqueeze(1))
    #     return series_list, prior1_list, prior2_list

    def forward(self, series, prior1, prior2, patch_size, attn_mask=None):
        for attn_layer in self.attn_layers:
            series, prior1, prior2 = attn_layer(series, prior1, prior2, patch_size, mask=attn_mask)
        return series, prior1, prior2


class DCdetector(nn.Module):
    def __init__(self, enc_in, c_out, n_heads=1, d_model=256, e_layers=3, patch_size=3, d_ff=512, dropout=0.0, activation='gelu', output_attention=True):
        super(DCdetector, self).__init__()
        self.output_attention = output_attention
        self.patch_size = patch_size

        self.embedding_window_size = DataEmbedding(enc_in, d_model, dropout)
         
        # Dual Attention Encoder
        self.encoder = Encoder(
            [
                EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        self.final_encoder = EncoderBlock(d_model, n_heads, d_ff, dropout)

        self.out= nn.Linear(d_model, c_out) 
        
        # 将RevIN层移动到初始化中，以便正确处理设备分配
        # 注意：这里我们使用一个占位符，实际的num_features会在第一次forward时设置
        self.revin_layer = None


    def forward(self, x):
        B, L, M = x.shape #Batch win_size channel
        
        # 如果RevIN层还没有初始化，或者特征数量发生变化，则重新创建
        if self.revin_layer is None or self.revin_layer.num_features != M:
            self.revin_layer = RevIN(num_features=M)
            # 确保RevIN层在正确的设备上
            if x.is_cuda:
                self.revin_layer = self.revin_layer.cuda()
        
        revin_layer = self.revin_layer
        x = revin_layer(x, 'norm')
        x = self.embedding_window_size(x)

        # Single patch size operation
        series, prior1, prior2 = self.encoder(x, x, x, self.patch_size)
        series, prior1, prior2 = self.final_encoder(series, prior1, prior2, self.patch_size, is_out=True)
        
        # Apply FFLayer to all outputs
        series = self.out(series)
        prior1 = self.out(prior1)
        prior2 = self.out(prior2)

        return series, prior1, prior2
        

