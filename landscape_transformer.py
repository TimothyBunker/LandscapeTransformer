import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import logging


# --- 1. Landscape Attention Module (Refined) ---
class LandscapeAttention(nn.Module):
    """
    Attends a query sequence to a set of learnable global prototypes.
    Outputs a context vector summarizing the query's relation to the prototype space.
    """
    def __init__(self, query_dim, num_prototypes, prototype_key_dim, prototype_value_dim):
        super().__init__()
        self.num_prototypes = num_prototypes
        self.query_dim = query_dim
        self.prototype_key_dim = prototype_key_dim
        self.prototype_value_dim = prototype_value_dim

        # Learnable prototypes used for matching queries
        self.prototype_keys = nn.Parameter(torch.randn(num_prototypes, prototype_key_dim))
        # Learnable values associated with prototypes, returned weighted by attention
        self.prototype_values = nn.Parameter(torch.randn(num_prototypes, prototype_value_dim))

        # Projection for the incoming query to match prototype key dimension
        self.query_proj = nn.Linear(query_dim, prototype_key_dim)
        self.scale = math.sqrt(prototype_key_dim)

        # Initialize parameters (optional but good practice)
        nn.init.xavier_uniform_(self.prototype_keys)
        nn.init.xavier_uniform_(self.prototype_values)
        if hasattr(self.query_proj, 'weight'):
            nn.init.xavier_uniform_(self.query_proj.weight)
        if self.query_proj.bias is not None:
            nn.init.zeros_(self.query_proj.bias)


    def forward(self, query):
        """
        Args:
            query (torch.Tensor): Input query tensor. Shape: [batch_size, seq_len, query_dim]

        Returns:
            torch.Tensor: Landscape context vector (K'). Shape: [batch_size, seq_len, prototype_value_dim]
        """
        batch_size, seq_len, _ = query.shape

        # Project query
        q_proj = self.query_proj(query) # [batch_size, seq_len, prototype_key_dim]

        # Calculate similarity (Scaled Dot-Product Attention)
        # q_proj: [b, s, pk_dim], self.prototype_keys.T: [pk_dim, num_proto]
        attn_scores = torch.matmul(q_proj, self.prototype_keys.t()) / self.scale
        # attn_scores shape: [batch_size, seq_len, num_prototypes]

        # Normalize scores across prototypes for each sequence position
        attn_weights = F.softmax(attn_scores, dim=-1)
        # attn_weights shape: [batch_size, seq_len, num_prototypes]

        # Calculate weighted sum of prototype values
        # attn_weights: [b, s, num_proto], self.prototype_values: [num_proto, pv_dim]
        landscape_context = torch.matmul(attn_weights, self.prototype_values)
        # landscape_context (K') shape: [batch_size, seq_len, prototype_value_dim]

        return landscape_context

# --- 3. Modified Transformer Block (Pre-Norm Architecture) - UPDATED ---
class LandscapeTransformerBlock(nn.Module):
    """
    Transformer Encoder Block using Pre-Norm architecture, augmented with
    Landscape Attention to incorporate global context before the FFN.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, num_prototypes, prototype_key_dim, prototype_value_dim, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # Standard Multi-Head Self-Attention
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # Landscape Attention Module
        self.landscape_attn = LandscapeAttention(embed_dim, num_prototypes, prototype_key_dim, prototype_value_dim)
        self.landscape_proj = nn.Linear(prototype_value_dim, embed_dim)

        # Standard Feed-Forward Network
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

        # Initialize projections (optional)
        if hasattr(self.landscape_proj, 'weight'):
             nn.init.xavier_uniform_(self.landscape_proj.weight)
        if self.landscape_proj.bias is not None:
            nn.init.zeros_(self.landscape_proj.bias)

    # UPDATED forward signature and self_attn call
    def forward(self, x, key_padding_mask=None):
        """
        Args:
            x (torch.Tensor): Input tensor. Shape: [batch_size, seq_len, embed_dim]
            key_padding_mask (torch.Tensor, optional): Mask for padding keys in self-attention.
                                                      Shape: [batch_size, seq_len]. True indicates padding.
                                                      Defaults to None.
        Returns:
            torch.Tensor: Output tensor. Shape: [batch_size, seq_len, embed_dim]
        """
        # --- Self-Attention Sub-layer (Pre-Norm) ---
        residual = x
        x_norm1 = self.norm1(x)
        attn_output, _ = self.self_attn(
            x_norm1, x_norm1, x_norm1,
            key_padding_mask=key_padding_mask, # Pass the padding mask here
            attn_mask=None, # attn_mask is for causal/lookahead, not needed for MLM encoder
            need_weights=False
        )
        x = residual + self.dropout(attn_output)

        # --- Landscape Attention & FFN Sub-layer ---
        residual = x
        x_norm2 = self.norm2(x)
        landscape_context = self.landscape_attn(x_norm2)
        landscape_info = self.landscape_proj(landscape_context)
        ffn_input = x_norm2 + self.dropout(landscape_info)
        ffn_output = self.ffn(ffn_input)
        x = residual + self.dropout(ffn_output)

        return x

# --- 4. Overall Language Model - UPDATED ---
class LandscapeTransformerLM(nn.Module):
    """
    Language Model using a stack of LandscapeTransformerBlocks.
    """
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers,
                 num_prototypes, prototype_key_dim, prototype_value_dim,
                 dropout=0.1, max_len=512, pad_token_id=0): # Added pad_token_id
        super().__init__()
        self.embed_dim = embed_dim
        self.pad_token_id = pad_token_id
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id) # Use padding_idx
        self.positional_embedding = nn.Parameter(torch.randn(1, max_len, embed_dim))
        self.embed_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            LandscapeTransformerBlock(
                embed_dim, num_heads, ff_dim,
                num_prototypes, prototype_key_dim, prototype_value_dim,
                dropout
            )
            for _ in range(num_layers)
        ])
        self.norm_out = nn.LayerNorm(embed_dim)
        self.output_head = nn.Linear(embed_dim, vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding) and module.padding_idx is not None:
            # Initialize non-padding embeddings
             nn.init.normal_(module.weight[:-1], mean=0.0, std=0.02)
             # Initialize padding embedding to zero
             nn.init.zeros_(module.weight[module.padding_idx])
        elif isinstance(module, nn.Embedding):
             nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)
        # Correct initialization for positional_embedding if it exists and is the current module
        if hasattr(self, 'positional_embedding') and isinstance(module, LandscapeTransformerLM):
            nn.init.normal_(self.positional_embedding, mean=0.0, std=0.02)


    # UPDATED forward signature
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids (torch.Tensor): Input token IDs. Shape: [batch_size, seq_len]
            attention_mask (torch.Tensor, optional): Mask indicating non-padding tokens (1) and padding (0).
                                                    Shape: [batch_size, seq_len]. Defaults to None.
        Returns:
            torch.Tensor: Output logits over vocabulary. Shape: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape

        # 1. Embeddings
        tok_emb = self.token_embedding(input_ids) * math.sqrt(self.embed_dim)
        pos_emb = self.positional_embedding[:, :seq_len, :]
        x = tok_emb + pos_emb
        x = self.embed_dropout(x)

        # --- Create key_padding_mask ---
        # True where padded (0 in original mask), False otherwise
        key_padding_mask = None
        if attention_mask is not None:
            if attention_mask.dim() == 2: # Expected [batch_size, seq_len]
                 key_padding_mask = (attention_mask == 0)
            else:
                 logging.warning(f"Unexpected attention_mask shape: {attention_mask.shape}. Expected [batch_size, seq_len]. Ignoring mask.")


        # 2. Transformer Blocks
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask) # Pass the correct mask

        # 3. Final Output
        x = self.norm_out(x)
        logits = self.output_head(x)

        return logits