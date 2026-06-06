
import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class FeatureProjection(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.25):
        super(FeatureProjection, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        return self.projection(x)


class MultiModalConcatModel(nn.Module):
    def __init__(
        self,
        path_embedding_dim=1536,
        text_embedding_dim=512, 
        hidden_dim=256,
        dropout=0.25,
        n_classes=2
    ):
        super(MultiModalConcatModel, self).__init__()

        self.path_embedding_dim = path_embedding_dim
        self.text_embedding_dim = text_embedding_dim
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes

        self.path_projection = FeatureProjection(
            input_dim=path_embedding_dim,
            output_dim=hidden_dim,
            dropout=dropout
        )


        self.text_projection = FeatureProjection(
            input_dim=text_embedding_dim,
            output_dim=hidden_dim,
            dropout=dropout
        )


        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, n_classes)
        )

        initialize_weights(self)

    def forward(self, path_features, text_features):
        # path_features: [B, N, C_path] -> [B, N, hidden_dim]
        path_proj = self.path_projection(path_features)

        # text_features: [B, C_text] -> [B, hidden_dim] -> [B, 1, hidden_dim]
        text_proj = self.text_projection(text_features)
        text_proj = text_proj.unsqueeze(1)  # [B, 1, hidden_dim]

        # path_proj: [B, N, hidden_dim]
        # text_proj: [B, 1, hidden_dim]
        concat_embed = torch.cat([path_proj, text_proj], dim=1)  # [B, N+1, hidden_dim]

        pooled = concat_embed.mean(dim=1)  # [B, hidden_dim]

        logits = self.classifier(pooled)  # [B, n_classes]

        return logits

    def get_embedding(self, path_features, text_features):
        path_proj = self.path_projection(path_features)
        text_proj = self.text_projection(text_features)
        text_proj = text_proj.unsqueeze(1)
        concat_embed = torch.cat([path_proj, text_proj], dim=1)
        pooled = concat_embed.mean(dim=1)
        return pooled

