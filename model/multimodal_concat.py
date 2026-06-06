
import torch
import torch.nn as nn


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class MultiModalConcatModel(nn.Module):
    def __init__(
        self,
        path_embedding_dim=1536,
        text_embedding_dim=512,
        gate=False,
        size_arg="big",
        dropout=False,
        n_classes=2,
        hidden_dim=512
    ):
        super(MultiModalConcatModel, self).__init__()

        self.path_embedding_dim = path_embedding_dim
        self.text_embedding_dim = text_embedding_dim
        self.n_classes = n_classes

        self.path_slide_dim = path_embedding_dim

        self.text_projection = nn.Sequential(
            nn.Linear(text_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25) if dropout else nn.Identity(),
        )

        concat_dim = self.path_slide_dim + hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.25) if dropout else nn.Identity(),
            nn.Linear(hidden_dim, n_classes)
        )

        initialize_weights(self)

    def forward(self, path_features, text_features):
        path_slide_embed = path_features.float().mean(dim=1)  # [B, path_embedding_dim]
        text_embed = self.text_projection(text_features)  # [B, hidden_dim]
        concat_embed = torch.cat([path_slide_embed, text_embed], dim=1)  # [B, concat_dim]
        logits = self.classifier(concat_embed)  # [B, n_classes]

        return logits

    def get_embedding(self, path_features, text_features):
        path_slide_embed = path_features.float().mean(dim=1)
        text_embed = self.text_projection(text_features)
        concat_embed = torch.cat([path_slide_embed, text_embed], dim=1)
        return concat_embed
