import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class PatchWiseMultiHeadAttention(nn.Module):
    """Patch-wise Multi-Head Attention mechanism"""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(PatchWiseMultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / sqrt(self.d_k)
        
    def forward(self, x, patch_size, mask=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
        Returns:
            output: Attention output of shape (batch_size, seq_len, d_model)
            attention_weights: Attention weights for visualization
        """
        batch_size, seq_len, d_model = x.shape
        
        # Reshape input into patches
        # x: (batch_size, seq_len, d_model) -> (batch_size, num_patches, patch_size, d_model)
        num_patches = seq_len // patch_size
        x_patches = x[:, :num_patches * patch_size, :].view(
            batch_size, num_patches, patch_size, d_model
        )
        
        # Apply linear transformations and reshape for multi-head attention
        # Shape: (batch_size, num_patches, patch_size, n_heads, d_k)
        q_patches = self.w_q(x_patches).view(batch_size, num_patches, patch_size, self.n_heads, self.d_k)
        k_patches = self.w_k(x_patches).view(batch_size, num_patches, patch_size, self.n_heads, self.d_k)
        v_patches = self.w_v(x_patches).view(batch_size, num_patches, patch_size, self.n_heads, self.d_k)
        
        # Transpose for attention computation
        # Shape: (batch_size, n_heads, num_patches, patch_size, d_k)
        q_patches = q_patches.transpose(2, 3).transpose(1, 2)
        k_patches = k_patches.transpose(2, 3).transpose(1, 2)
        v_patches = v_patches.transpose(2, 3).transpose(1, 2)
        
        # Compute patch-wise attention
        # Inter-patch attention: attention between different patches
        q_inter = q_patches.mean(dim=3)  # (batch_size, n_heads, num_patches, d_k)
        k_inter = k_patches.mean(dim=3)  # (batch_size, n_heads, num_patches, d_k)
        v_inter = v_patches.mean(dim=3)  # (batch_size, n_heads, num_patches, d_k)
        
        scores_inter = torch.matmul(q_inter, k_inter.transpose(-2, -1)) * self.scale
        attn_inter = F.softmax(scores_inter, dim=-1)
        attn_inter = self.dropout(attn_inter)
        
        # Intra-patch attention: attention within each patch
        scores_intra = torch.matmul(q_patches, k_patches.transpose(-2, -1)) * self.scale
        attn_intra = F.softmax(scores_intra, dim=-1)
        attn_intra = self.dropout(attn_intra)
        
        # Apply attention
        # Fix inter-patch attention application
        out_inter = torch.matmul(attn_inter, v_inter)  # (batch_size, n_heads, num_patches, d_k)
        out_inter = out_inter.unsqueeze(3).expand(-1, -1, -1, patch_size, -1)  # Expand to patch_size
        out_intra = torch.matmul(attn_intra, v_patches)
        
        # Combine inter and intra patch attention
        out = out_inter + out_intra
        
        # Reshape back to original format
        # (batch_size, n_heads, num_patches, patch_size, d_k) -> (batch_size, seq_len, d_model)
        out = out.transpose(1, 2).transpose(2, 3).contiguous().view(
            batch_size, num_patches * patch_size, d_model
        )
        
        # Apply output projection
        output = self.w_o(out)
        
        # Handle remaining sequence if seq_len is not divisible by patch_size
        if seq_len > num_patches * patch_size:
            remaining = x[:, num_patches * patch_size:, :]
            output = torch.cat([output, remaining], dim=1)
        
        return output, attn_inter


class PatchWiseFeedForward(nn.Module):
    """Patch-wise Feed Forward Network"""
    
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PatchWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))