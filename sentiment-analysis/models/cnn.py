import torch.nn as nn
import torch

class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv = nn.Conv1d(in_channels=embed_dim, out_channels=100, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(100, num_classes)

    def forward(self, input_ids):
        x = self.embedding(input_ids).permute(0, 2, 1)
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)
