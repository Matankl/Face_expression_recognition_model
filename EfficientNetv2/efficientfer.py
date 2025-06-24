import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights

class SelfAttention(nn.Module):
    """
    Self Attention Module
    """
    def __init__(self, in_channels, head_dim=64, heads=8):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(in_channels, head_dim * heads * 3)
        self.proj = nn.Linear(head_dim * heads, in_channels)
        
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.view(b, c, -1).permute(0, 2, 1) 
        
        qkv = self.qkv(x).chunk(3, dim=-1) 
        q, k, v = map(lambda t: t.view(b, -1, self.heads, self.head_dim).transpose(1, 2), qkv)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale 
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)  
        out = out.transpose(1, 2).reshape(b, -1, self.heads * self.head_dim)  
        out = self.proj(out)  
        
        out = out.permute(0, 2, 1).view(b, c, h, w)
        return out


class ChannelAttention(nn.Module):
    """
    Channel Attention Module for CBAM
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module for CBAM
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    """
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class EfficientFER(nn.Module):
    def __init__(self, num_classes=7, dropout_rate=0.3):
        super(EfficientFER, self).__init__()
        
        self.backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.DEFAULT)
        
        self.backbone.classifier = nn.Identity()
        
        self.self_attention = SelfAttention(1280)
        
        self.cbam = CBAM(1280)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        
        b, c = features.shape
        features_4d = features.view(b, c, 1, 1)
        
        sa_features = self.self_attention(features_4d)
        
        cbam_features = self.cbam(features_4d)
        
        combined_features = sa_features + cbam_features
        
        combined_features = self.adaptive_pool(combined_features).view(b, c)
        
        logits = self.classifier(combined_features)
        
        return logits