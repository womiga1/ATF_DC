import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules import ATTNLayer, FFLayer, FFTLayer, WPCLayer, MFLayer, WTCLayer

class EncoderBlock(nn.Module):
    """Single Patch-wise Transformer Encoder Block"""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super(EncoderBlock, self).__init__()
        
        self.attention = ATTNLayer(d_model, n_heads, dropout)
        # self.fft = FFTLayer(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=1, padding="same")
        self.fft = FFTLayer(in_channels=d_model, out_channels=d_model)
        # self.wpc = WTCLayer(in_channels=d_model, out_channels=d_model//2, kernel_size=4, stride=1)
        self.wpc = WTCLayer(in_channels=d_model, out_channels=d_model, kernel_size=4, stride=1)

        self.feed_forward_1 = FFLayer(d_model, d_ff, dropout)
        self.feed_forward_2 = FFLayer(d_model, d_ff, dropout)
        self.feed_forward_3 = FFLayer(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)

        self.mf = MFLayer(d_model)
        
    def forward(self, x1, x2, x3, patch_size, mask=None, is_out=False):
        # Multi-head attention with residual connection and layer norm
        attn_out, _ = self.attention(self.norm1(x1), patch_size, mask)
        attn_out = x1 + self.dropout(attn_out)
        ff_out1 = self.feed_forward_1(self.norm2(attn_out))
        x1 = attn_out + self.dropout(ff_out1)

        # FFT with residual connection and layer norm
        # fft_out = self.fft(self.norm1(x2).permute(0, 2, 1)).permute(0, 2, 1)
        # fft_out = x2 + self.dropout(fft_out)
        # ff_out2 = self.feed_forward_2(self.norm2(fft_out))
        # x2 = fft_out + self.dropout(ff_out2)
        x2 = self.fft(self.norm1(x2).permute(0, 2, 1)).permute(0, 2, 1)


        # WPC with residual connection and layer norm
        # wpc_out = self.wpc(self.norm1(x3).permute(0, 2, 1))
        # wpc_out = F.adaptive_max_pool1d(wpc_out, x3.shape[1]).permute(0, 2, 1)
        # wpc_out = x3 + self.dropout(wpc_out)
        # ff_out3 = self.feed_forward_3(self.norm2(wpc_out))
        # x3 = wpc_out + self.dropout(ff_out3)
        wpc_out = self.wpc(self.norm1(x3).permute(0, 2, 1))
        x3 = F.adaptive_avg_pool1d(wpc_out, x3.shape[1]).permute(0, 2, 1)

        if is_out:
            return x1, x2, x3
        x1_out, x2_out, x3_out = self.mf(x1.permute(0, 2, 1), x2.permute(0, 2, 1), x3.permute(0, 2, 1)) 
        x1_out = x1_out.permute(0, 2, 1)
        x2_out = x2_out.permute(0, 2, 1)
        x3_out = x3_out.permute(0, 2, 1)

        return x1_out, x2_out, x3_out
