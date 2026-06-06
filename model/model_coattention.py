
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, einsum


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


class CoAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.25):
        super(CoAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.to_q_a = nn.Linear(hidden_dim, hidden_dim)
        self.to_k_b = nn.Linear(hidden_dim, hidden_dim)
        self.to_v_b = nn.Linear(hidden_dim, hidden_dim)

        self.to_q_b = nn.Linear(hidden_dim, hidden_dim)
        self.to_k_a = nn.Linear(hidden_dim, hidden_dim)
        self.to_v_a = nn.Linear(hidden_dim, hidden_dim)

        self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim * 2)

    def forward(self, text_feat, path_feat):
        B, N, C = path_feat.shape
        text_feat = text_feat.unsqueeze(1)  # [B, 1, C]

        q_text = self.to_q_a(text_feat)  # [B, 1, hidden_dim]
        k_path = self.to_k_b(path_feat)  # [B, N, hidden_dim]
        v_path = self.to_v_b(path_feat)  # [B, N, hidden_dim]

        # Multi-head attention
        q_text = q_text.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, 1, head_dim]
        k_path = k_path.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)   # [B, heads, N, head_dim]
        v_path = v_path.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)   # [B, heads, N, head_dim]

        scores_text = torch.matmul(q_text, k_path.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_text = F.softmax(scores_text, dim=-1)
        text_output = torch.matmul(attn_text, v_path)  # [B, heads, 1, head_dim]
        text_output = text_output.transpose(1, 2).contiguous().view(B, 1, self.hidden_dim)  # [B, 1, hidden_dim]
        text_output = text_output.squeeze(1)  # [B, hidden_dim]

        # ========== Path as query, Text as key/value ==========
        q_path = self.to_q_b(path_feat)  # [B, N, hidden_dim]
        k_text = self.to_k_a(text_feat)  # [B, 1, hidden_dim]
        v_text = self.to_v_a(text_feat)  # [B, 1, hidden_dim]

        # Multi-head attention
        q_path = q_path.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, heads, N, head_dim]
        k_text = k_text.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)   # [B, heads, 1, head_dim]
        v_text = v_text.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)   # [B, heads, 1, head_dim]

        scores_path = torch.matmul(q_path, k_text.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_path = F.softmax(scores_path, dim=-1)
        path_output = torch.matmul(attn_path, v_text)  # [B, heads, N, head_dim]
        path_output = path_output.transpose(1, 2).contiguous().view(B, N, self.hidden_dim)  # [B, N, hidden_dim]

        path_output = path_output.mean(dim=1)  # [B, hidden_dim]

        return text_output, path_output


class MultiModalCoAttentionModel(nn.Module):
    def __init__(
        self,
        path_embedding_dim=1536,
        text_embedding_dim=512,
        hidden_dim=256,
        num_heads=4,
        dropout=0.25,
        n_classes=2
    ):
        super(MultiModalCoAttentionModel, self).__init__()

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


        self.co_attention = CoAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        concat_dim = hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, n_classes)
        )

        initialize_weights(self)

    def forward(self, path_features, text_features):

        path_proj = self.path_projection(path_features)  # [B, N, hidden_dim]

        text_proj = self.text_projection(text_features)  # [B, hidden_dim]


        text_output, path_output = self.co_attention(text_proj, path_proj)
        # text_output: [B, hidden_dim]
        # path_output: [B, hidden_dim]

        concat_embed = torch.cat([text_output, path_output], dim=1)  # [B, hidden_dim*2]

        logits = self.classifier(concat_embed)  # [B, n_classes]

        return logits

    def get_embedding(self, path_features, text_features):
        path_proj = self.path_projection(path_features)
        text_proj = self.text_projection(text_features)
        text_output, path_output = self.co_attention(text_proj, path_proj)
        concat_embed = torch.cat([text_output, path_output], dim=1)
        return concat_embed


