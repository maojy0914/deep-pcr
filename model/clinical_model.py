
import torch
import torch.nn as nn


class ClinicalMLP(nn.Module):
    def __init__(
        self,
        input_dim=512,
        hidden_dim=256,
        n_classes=2,
        dropout=0.25
    ):
        super(ClinicalMLP, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes)
        )

    def forward(self, x):
        logits = self.classifier(x)
        return logits

