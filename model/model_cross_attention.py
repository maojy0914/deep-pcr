
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


class CrossAttentionModule(nn.Module):
    def __init__(self, hidden_dim=256, num_heads=4):
        super(CrossAttentionModule, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.head_dim = hidden_dim // num_heads

        self.text_proj = nn.Linear(hidden_dim, hidden_dim)
        self.path_key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.path_value_proj = nn.Linear(hidden_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, path_features, text_features):
        B, N, C = path_features.shape  # [B, N, hidden_dim]

        # Text features: [B, C] -> [B, 1, hidden_dim]
        Q = self.text_proj(text_features).unsqueeze(1)

        # Path features: [B, N, hidden_dim] -> [B, N, hidden_dim]
        K = self.path_key_proj(path_features)

        # Path features: [B, N, hidden_dim] -> [B, N, hidden_dim]
        V = self.path_value_proj(path_features)

        # Q: [B, 1, hidden_dim] -> [B, 1, num_heads, head_dim] -> [B, num_heads, 1, head_dim]
        Q = Q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Q: [B, num_heads, 1, head_dim], K: [B, num_heads, head_dim, N]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)

        # Attention weights: [B, num_heads, 1, N]
        attn_weights = F.softmax(scores, dim=-1)

        # Apply attention to values
        # attn_weights: [B, num_heads, 1, N], V: [B, num_heads, N, head_dim]
        context = torch.matmul(attn_weights, V)  # [B, num_heads, 1, head_dim]

        context = context.transpose(1, 2).contiguous()
        context = context.view(B, 1, self.hidden_dim)
        context = context.squeeze(1)  # [B, hidden_dim]

        context = self.out_proj(context)

        attn_weights_agg = attn_weights.mean(dim=1).squeeze(1)  # [B, N]

        return context, attn_weights_agg


class MultiModalCrossAttentionModel(nn.Module):
    def __init__(
        self,
        path_embedding_dim=1536,
        text_embedding_dim=512,
        hidden_dim=256,
        num_heads=4,
        gate=False,
        dropout=0.25,
        n_classes=2
    ):
        super(MultiModalCrossAttentionModel, self).__init__()

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

        self.cross_attention = CrossAttentionModule(
            hidden_dim=hidden_dim,
            num_heads=num_heads
        )


        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, n_classes)
        )

        initialize_weights(self)

    def forward(self, path_features, text_features):

        path_proj = self.path_projection(path_features)  # [B, N, hidden_dim]
        text_proj = self.text_projection(text_features)  # [B, hidden_dim]
        cross_output, _ = self.cross_attention(path_proj, text_proj)  # [B, hidden_dim]

        logits = self.classifier(cross_output)  # [B, n_classes]

        return logits

    def get_embedding(self, path_features, text_features):
        path_proj = self.path_projection(path_features)
        text_proj = self.text_projection(text_features)
        cross_output, _ = self.cross_attention(path_proj, text_proj)
        return cross_output
