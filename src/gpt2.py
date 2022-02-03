import torch
from torch import einsum
from torch import nn 
from einops import rearrange, reduce, repeat
import math
import random
from collections import OrderedDict
import transformers
import torchtext
from tqdm import tqdm
import matplotlib.pyplot as plt

from gpt2_output import GPT2Output


class UnidirectionalMultiheadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads 
        self.head_size = hidden_size // num_heads
        assert self.head_size * num_heads == hidden_size
        self.attentionLL = nn.Linear(hidden_size, num_heads*self.head_size*3) 
        self.outputLL = nn.Linear(num_heads*self.head_size, hidden_size)
    
    def forward(self, x: torch.Tensor): # [batch, seq_length, hidden_size]
        # Shape: batch seq_len hidden_size*3
        KQV = self.attentionLL(x)
        KQV = rearrange(KQV, "batch seq_len (three num_heads head_size) -> batch num_heads seq_len head_size three ", num_heads=self.num_heads, three=3)
        Q = KQV[:, :, :, :, 0]
        K = KQV[:, :, :, :, 1]
        V = KQV[:, :, :, :, 2]
        # Multiplying K and Q
        attention_pattern = einsum('b n s h, b n t h -> b n s t', K, Q)
        # Scale
        attention_pattern = attention_pattern / math.sqrt(self.head_size)
        # Key (row) must be less than Query (col), if not we set it to 1e-4
        attention_pattern = torch.triu(attention_pattern) + (-1e4) * torch.tril(torch.ones_like(attention_pattern), diagonal=-1)        
        # Softmax: batch num_heads key_len query_len, so we want to softmax over the keys
        #  so dim=2
        attention_pattern = torch.nn.Softmax(dim=2)(attention_pattern)
        # Multiply by V
        out = einsum('b n k q, b n k h -> b n q h', attention_pattern, V)
        out = rearrange(out, 'batch num_heads seq_len head_size -> batch seq_len (num_heads head_size)')
        out = self.outputLL(out) 
        return out


class GPT2Block(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, 
                dropout: float, layer_norm_epsilon: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.attn = UnidirectionalMultiheadAttention(hidden_size, num_heads)
        self.ln2 = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln1(x))
        x = x + self.dropout(self.linear2(torch.nn.functional.gelu(self.linear1(self.ln2(x)))))
        return x 


class GPT2(nn.Module):
    def __init__(self, num_layers, num_heads, vocab_size,
                hidden_size, max_position_embeddings, dropout, 
                layer_norm_epsilon):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.GPTBlocks = nn.Sequential(
            *[GPT2Block(hidden_size, num_heads, dropout, layer_norm_epsilon) 
                for i in range(num_layers)]
        )
        self.last_token_encodings = None
        self.layer_norm = nn.LayerNorm(hidden_size, layer_norm_epsilon)

    def forward(self, input_ids): # [batch, seq_len]
        tokens = self.token_embedding(input_ids)
        batch, seq_len = input_ids.shape
        position_ids = repeat(torch.arange(seq_len), 's -> b s', b = batch) 
        positions = self.position_embedding(position_ids)
        embedding = tokens + positions
        x = self.dropout(embedding)
        x = self.GPTBlocks(x)
        self.last_token_encodings = x
        final_encodings = self.layer_norm(x)[:,-1,:]
        logits = einsum('b c, v c -> b v', final_encodings, self.token_embedding.weight)
        return GPT2Output(logits, final_encodings)
