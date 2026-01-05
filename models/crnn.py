import torch.nn as nn

try:
    from .fusion import AttentionFusion
except ImportError:
    from fusion import AttentionFusion


class MultiFrameCRNN(nn.Module):
    """
    Multi-frame CRNN for license plate recognition.
    
    Architecture:
        CNN backbone -> AttentionFusion -> BiLSTM -> FC -> LogSoftmax
    
    Takes 5 frames as input and outputs character predictions.
    """
    
    def __init__(self, num_classes, hidden_size=256):
        """
        Args:
            num_classes: Number of output classes (characters + blank)
            hidden_size: Hidden size for LSTM layers
        """
        super().__init__()
        
        # CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True)
        )
        
        # Attention-based temporal fusion
        self.fusion = AttentionFusion(channels=512)
        
        # Bidirectional LSTM
        self.rnn = nn.Sequential(
            nn.LSTM(512, hidden_size, bidirectional=True, batch_first=True, num_layers=2, dropout=0.25)
        )
        
        # Output classifier
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [B, 5, C, H, W]
        
        Returns:
            Log probabilities of shape [B, T, num_classes]
        """
        b, t, c, h, w = x.size()
        
        # Process all frames through CNN
        x = x.view(b * t, c, h, w)
        feat = self.cnn(x)
        
        # Fuse features across frames
        fused = self.fusion(feat)
        
        # RNN sequence modeling
        out = self.fc(self.rnn(fused.permute(0, 2, 1))[0])
        
        return out.log_softmax(2)
