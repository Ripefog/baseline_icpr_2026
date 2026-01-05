import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionFusion(nn.Module):
    """
    Attention-based temporal fusion module.
    
    Learns attention weights for each frame and produces a weighted combination
    of features from multiple frames.
    """
    
    def __init__(self, channels):
        """
        Args:
            channels: Number of input feature channels
        """
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(channels // 8, 1, kernel_size=1)
        )
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B*5, C, H, W] (5 frames per sample)
        
        Returns:
            Fused tensor of shape [B, C, W]
        """
        b_frames, c, h, w = x.size()
        b_size = b_frames // 5
        
        # Reshape for frame-wise processing
        x_view = x.view(b_size, 5, c, w)
        
        # Compute attention scores
        scores = self.score_net(x).view(b_size, 5, 1, w)
        
        # Apply softmax and weighted sum
        return torch.sum(x_view * F.softmax(scores, dim=1), dim=1)
