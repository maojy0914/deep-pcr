
import os
import pandas as pd
import torch
from tqdm import tqdm

import conch.open_clip_custom as occ


def load_text_encoder_and_tokenizer(checkpoint_path):
    tokenizer = occ.get_tokenizer()

    model = create_text_encoder_from_pretrained(checkpoint_path)
    model.eval()

    return model, tokenizer


def create_text_encoder_from_pretrained(checkpoint_path):
    # restore checkpoint
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    text_state_dict = {}
    for key in state_dict.keys():
        if key.startswith('text.'):
            new_key = key[5:]
            text_state_dict[new_key] = state_dict[key]


    vocab_size = text_state_dict['token_embedding.weight'].shape[0]
    embed_dim = text_state_dict['token_embedding.weight'].shape[1]


    layer_indices = set()
    for key in text_state_dict.keys():
        if key.startswith('transformer.resblocks.'):
            parts = key.replace('transformer.resblocks.', '').split('.')
            layer_indices.add(int(parts[0]))
    num_layers = len(layer_indices)

    output_dim = text_state_dict['text_projection'].shape[1]
    out_proj_weight = text_state_dict['transformer.resblocks.0.attn.out_proj.weight']
    num_heads = out_proj_weight.shape[0] // embed_dim

    max_seq_len = text_state_dict['positional_embedding'].shape[0]


    model = TextEncoder(embed_dim=embed_dim, output_dim=output_dim, vocab_size=vocab_size,
                        num_heads=num_heads, layers=num_layers, max_seq_len=max_seq_len)


    mapped_state_dict = {}
    for key, value in text_state_dict.items():
        if key.startswith('transformer.resblocks.'):
            new_key = key.replace('transformer.resblocks.', 'transformer.')
            new_key = new_key.replace('.mlp.c_fc.', '.mlp.0.')
            new_key = new_key.replace('.mlp.c_proj.', '.mlp.2.')
            mapped_state_dict[new_key] = value
        elif key == 'cls_emb':
            mapped_state_dict['cls_token'] = value.unsqueeze(0).unsqueeze(0)
        elif key == 'text_projection':
            mapped_state_dict[key] = value
        else:
            mapped_state_dict[key] = value

    missing, unexpected = model.load_state_dict(mapped_state_dict, strict=False)

    return model


class TextEncoder(torch.nn.Module):

    def __init__(self, embed_dim=768, output_dim=512, vocab_size=32007, num_heads=12, layers=12, max_seq_len=128):
        super(TextEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.layers = layers

        # Token embedding
        self.token_embedding = torch.nn.Embedding(vocab_size, embed_dim)

        # Positional embedding
        self.positional_embedding = torch.nn.Parameter(torch.randn(max_seq_len, embed_dim) * 0.02)

        # CLS token
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Transformer blocks
        self.transformer = torch.nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(layers)
        ])

        # Layer norm
        self.ln_final = torch.nn.LayerNorm(embed_dim)

        # Output projection
        self.text_projection = torch.nn.Linear(embed_dim, output_dim)

    def forward(self, text_tokens):
        """text_tokens: [B, L] token IDs"""
        B, L = text_tokens.shape

        # Token embedding
        x = self.token_embedding(text_tokens)

        # Add positional embedding
        if L <= self.positional_embedding.shape[0]:
            x = x + self.positional_embedding[:L].unsqueeze(0)
        else:
            x = x + self.positional_embedding[:L].unsqueeze(0)

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)

        for blk in self.transformer:
            x = blk(x)

        x = self.ln_final(x)
        cls_output = x[:, 0]
        output = self.text_projection(cls_output)

        return output


class TransformerBlock(torch.nn.Module):
    """Transformer block for text encoder"""

    def __init__(self, embed_dim, num_heads):
        super(TransformerBlock, self).__init__()
        self.ln_1 = torch.nn.LayerNorm(embed_dim)
        self.attn = torch.nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln_2 = torch.nn.LayerNorm(embed_dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(embed_dim, embed_dim * 4),
            torch.nn.GELU(),
            torch.nn.Linear(embed_dim * 4, embed_dim)
        )

    def forward(self, x):
        attn_output, _ = self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x))
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return x


def extract_text_embedding(model, tokenizer, text):
    tokens = occ.tokenize(tokenizer, [text])
    with torch.no_grad():
        embedding = model(tokens)
    return embedding.squeeze(0)


def main():

    prompt_csv = ""
    output_dir = ""
    checkpoint_path = ""

    os.makedirs(output_dir, exist_ok=True)


    print("Loading CONCH v1.5 text encoder...")
    model, tokenizer = load_text_encoder_and_tokenizer(checkpoint_path)
    print("Model loaded! Embedding dimension: 512")


    df = pd.read_csv(prompt_csv)
    print("Loaded %d samples from %s" % (len(df), prompt_csv))

    template_cols = [col for col in df.columns if col != 'slide_id']

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        slide_id = row['slide_id']
        text_parts = []
        for col in template_cols:
            text = str(row[col]).strip()
            if text and text.lower() != 'nan':
                text_parts.append(text)

        full_text = " ".join(text_parts)

        embedding = extract_text_embedding(model, tokenizer, full_text)

        output_path = os.path.join(output_dir, "%s.pt" % slide_id)
        torch.save(embedding, output_path)

    print("\nDone! Embeddings saved to %s" % output_dir)
    print("Total embeddings: %d" % len(df))
    print("Embedding dimension: 512")


if __name__ == "__main__":
    main()