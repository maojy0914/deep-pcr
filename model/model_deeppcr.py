
import torch
import torch.nn as nn
import torch.nn.functional as F
from nystrom_attention import NystromAttention


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
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


class Attn_Net(nn.Module):
    def __init__(self, L=512, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [
            nn.Linear(L, D),
            nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [
            nn.Linear(L, D),
            nn.Tanh()]

        self.attention_b = [nn.Linear(L, D),
                            nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)
        return A, x


class AttentionMIL(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, gate=False, dropout=0.25):
        super(AttentionMIL, self).__init__()

        if dropout > 0:
            self.fc = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU()
            )

        if gate:
            self.attention_net = Attn_Net_Gated(L=hidden_dim, D=hidden_dim // 2, dropout=dropout, n_classes=1)
        else:
            self.attention_net = Attn_Net(L=hidden_dim, D=hidden_dim // 2, dropout=dropout, n_classes=1)

        initialize_weights(self)

    def forward(self, h):
        h = h.float()

        if h.dim() == 3:
            B, N, C = h.shape
            h = h.view(-1, C)
        elif h.dim() == 2:
            B, N = 1, h.shape[0]
        else:
            raise ValueError("AttentionMIL expects a 2D or 3D tensor")

        A, h = self.attention_net(h)

        A = A.view(B, N, -1).transpose(1, 2)
        A = F.softmax(A, dim=2)
        h = h.view(B, N, -1)

        M = torch.bmm(A, h).squeeze(1)

        return M


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_ratio=4, dropout=0.25):
        super(FeedForward, self).__init__()
        hidden_dim = dim * mlp_ratio
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

    def forward(self, x):
        return self.net(x)


class NystromTransformerBlock(nn.Module):
    def __init__(self, dim=256, num_heads=8, dropout=0.25, mlp_ratio=4):
        super(NystromTransformerBlock, self).__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.norm1 = nn.LayerNorm(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // num_heads,
            heads=num_heads,
            num_landmarks=max(1, dim // 2),
            pinv_iterations=6,
            residual=True,
            dropout=dropout
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim=dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x):
        y = x + self.attn(self.norm1(x))
        z = y + self.mlp(self.norm2(y))
        return z


class NativeTransformerBlock(nn.Module):
    def __init__(self, dim=256, num_heads=8, dropout=0.25, mlp_ratio=4):
        super(NativeTransformerBlock, self).__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim=dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(self, x):
        norm_x = self.norm1(x)
        norm_x = norm_x.transpose(0, 1)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, need_weights=False)
        attn_out = attn_out.transpose(0, 1)
        y = x + attn_out
        z = y + self.mlp(self.norm2(y))
        return z


class DeepPCRModel(nn.Module):
    def __init__(
        self,
        path_embedding_dim=1024,
        text_embedding_dim=512,
        hidden_dim=256,
        num_heads=8,
        gate=False,
        dropout=0.25,
        n_classes=2
    ):
        super(DeepPCRModel, self).__init__()

        self.path_embedding_dim = path_embedding_dim
        self.text_embedding_dim = text_embedding_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.n_classes = n_classes

        self.path_projection = nn.Sequential(
            nn.Linear(path_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

        self.text_projection = nn.Sequential(
            nn.Linear(text_embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        )

        self.path_transformer = NystromTransformerBlock(
            dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.text_transformer = NativeTransformerBlock(
            dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.attention_mil = AttentionMIL(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            gate=gate,
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
        path_proj = self.path_projection(path_features)
        path_embed = self.path_transformer(path_proj)

        text_embed = self.text_projection(text_features)
        if text_embed.dim() == 2:
            text_embed = text_embed.unsqueeze(1)
        text_embed = self.text_transformer(text_embed)

        multimodal_tokens = torch.cat([path_embed, text_embed], dim=1)
        embed = self.attention_mil(multimodal_tokens)  # [B, hidden_dim]

        logits = self.classifier(embed)  # [B, n_classes]

        return logits

    def get_embedding(self, path_features, text_features):
        path_proj = self.path_projection(path_features)
        path_embed = self.path_transformer(path_proj)

        text_embed = self.text_projection(text_features)
        if text_embed.dim() == 2:
            text_embed = text_embed.unsqueeze(1)
        text_embed = self.text_transformer(text_embed)

        multimodal_tokens = torch.cat([path_embed, text_embed], dim=1)
        embed = self.attention_mil(multimodal_tokens)
        return embed
