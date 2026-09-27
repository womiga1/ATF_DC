import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class WPConv(nn.Module):
    def __init__(self, in_channels=1, out_channels=16, kernel_size=4, stride=2):
        super(WPConv, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.out_channels = out_channels
        
        self.h = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size))
        
        sqrt3 = math.sqrt(3)
        sqrt2 = math.sqrt(2)
        h_db2 = [
            (1 + sqrt3) / (4 * sqrt2),
            (3 + sqrt3) / (4 * sqrt2),
            (3 - sqrt3) / (4 * sqrt2),
            (1 - sqrt3) / (4 * sqrt2)
        ]
        h_db2_tensor = torch.FloatTensor(h_db2)

        with torch.no_grad():
            self.h.data = h_db2_tensor.repeat(out_channels, in_channels, 1)

    def generate_g(self, h):
        reversed_h = torch.flip(h, dims=[-1])
        n = torch.arange(self.kernel_size, device=h.device)
        sign = (-1) ** n
        sign = sign.view(1, 1, -1)
        g = reversed_h * sign
        return g
    
    def forward(self, x):
        g = self.generate_g(self.h)

        padding = 2

        y_a = F.conv1d(x, self.h, stride=self.stride, padding=padding)
        y_d = F.conv1d(x, g, stride=self.stride, padding=padding)
        
        y = torch.cat([y_a, y_d], dim=1)
        
        return y
    
    def wavelet_regularizer(self):
        reg = 0.0
        for c in range(self.out_channels):
            h_c = self.h[c, 0, :]
            
            term1 = torch.abs(torch.sum(h_c**2) - 1)
            
            term2 = 0.0
            for k in range(1, (self.kernel_size//2)+1):
                corr = torch.sum(h_c[:-2*k] * h_c[2*k:])
                term2 += torch.abs(corr)

            term3 = torch.abs(torch.sum(h_c) - torch.sqrt(torch.FloatTensor([2.0])))
            
            reg += term1 + term2 + term3
        
        return reg / self.out_channels
    

class InvWPConv(nn.Module):
    def __init__(self, wpconv_layer):
        super().__init__()
        self.stride = wpconv_layer.stride
        self.kernel_size = wpconv_layer.kernel_size
        self.h = wpconv_layer.h
    
    def generate_g(self, h):
        reversed_h = torch.flip(h, dims=[-1])
        n = torch.arange(self.kernel_size, device=h.device)
        sign = (-1) ** n
        return reversed_h * sign.view(1, 1, -1)
    
    def forward(self, x):
        x_a, x_d = torch.chunk(x, 2, dim=1)
        
        h_t = torch.flip(self.h, dims=[-1])
        g_t = self.generate_g(h_t)

        output_padding = (x_a.shape[-1] * 2 - 1) - (x_a.shape[-1] -1)*2 - (self.kernel_size - 2)
        output_padding = max(0, output_padding)

        y_a = F.conv_transpose1d(x_a, h_t, stride=self.stride, padding=1, output_padding=output_padding)
        y_d = F.conv_transpose1d(x_d, g_t, stride=self.stride, padding=1, output_padding=output_padding)
        return y_a + y_d
    
class AdaptiveSoftshrink(nn.Module):
    def __init__(self, alpha_init=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.FloatTensor([alpha_init]))
        
    def forward(self, x):
        noise_std = torch.mean(torch.abs(x), dim=(1,2), keepdim=True)
        threshold = self.alpha * noise_std
        return torch.where(
            torch.abs(x) > threshold,
            x - torch.sign(x) * threshold,
            torch.zeros_like(x)
        )