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
    
    def forward(self, x: torch.Tensor, past_key_values = None, return_key_values = False): # [batch, seq_length, hidden_size]
        if past_key_values is None:
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
            if return_key_values:
                assert x.shape[0] == 1
                return out, torch.cat((K,V), dim = 3) 
            else:
                return out
        else:
            assert x.shape == (1,1,self.hidden_size)
            kqv = self.attentionLL(x)
            kqv = rearrange(kqv, "batch seq_len (three num_heads head_size) -> batch num_heads seq_len head_size three ", num_heads=self.num_heads, three=3)
            q = kqv[0, :, :, :, 0]
            k = kqv[0, :, :, :, 1]
            v = kqv[0, :, :, :, 2]
            oldK, oldV = torch.split(past_key_values, (self.head_size, self.head_size), dim = 2)
            K = torch.cat((oldK, k), dim = 1)
            V = torch.cat((oldV, v), dim = 1)
            attention_pattern = einsum('n s h, n t h -> n s t', q, K)
            attention_pattern = attention_pattern / math.sqrt(self.head_size)
            attention_pattern = torch.nn.Softmax(dim=2)(attention_pattern)
            out = einsum('n s t, n t h -> n s h', attention_pattern, V)
            out = rearrange(out, '(batch num_heads) seq_len head_size -> batch seq_len (num_heads head_size)', batch = 1)
            out = self.outputLL(out)
            if return_key_values:
                return out, torch.cat((k,v), dim = 2).unsqueeze(0)
            else:
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

    def forward(self, x: torch.Tensor, past_key_values = None, return_key_values = False):
        if return_key_values:
            res = x
            x, keyvals = self.attn(self.ln1(x), past_key_values=past_key_values, return_key_values = True)
            x = x + res
            x = x + self.dropout(self.linear2(torch.nn.functional.gelu(self.linear1(self.ln2(x)))))
            return x, keyvals
        else:
            x = x + self.attn(self.ln1(x), past_key_values=past_key_values, return_key_values = False)
            x = x + self.dropout(self.linear2(torch.nn.functional.gelu(self.linear1(self.ln2(x)))))
            return x


class GPT2(nn.Module):
    def __init__(self, num_layers, num_heads, vocab_size,
                hidden_size, max_position_embeddings, dropout, 
                layer_norm_epsilon, use_cache=False):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_position_embeddings, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.GPTBlocks = nn.Sequential(
            *[GPT2Block(hidden_size, num_heads, dropout, layer_norm_epsilon) 
                for i in range(num_layers)]
        )
        self.layer_norm = nn.LayerNorm(hidden_size, layer_norm_epsilon)
        self.use_cache = use_cache
        self.head_size = hidden_size // num_heads
        self.past_key_values = torch.zeros((num_layers, num_heads, 0, 2*self.head_size))

    def forward(self, input_ids): # [batch, seq_len]
        if not self.use_cache:
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
        else:
            if self.past_key_values.shape[2] == 0:
                tokens = self.token_embedding(input_ids)
                batch, seq_len = input_ids.shape
                position_ids = repeat(torch.arange(seq_len), 's -> b s', b = batch) 
                positions = self.position_embedding(position_ids)
                embedding = tokens + positions
                x = self.dropout(embedding)
                new_key_values = []
                for gptblock in self.GPTBlocks:
                    x, new_key_value = gptblock(x, return_key_values = True)
                    new_key_values.append(new_key_value)
                self.past_key_values = torch.cat(new_key_values, dim=0)
                final_encodings = self.layer_norm(x)[:,-1,:]
                logits = einsum('b c, v c -> b v', final_encodings, self.token_embedding.weight)
                return GPT2Output(logits, final_encodings)
            else:
                tokens = self.token_embedding(input_ids[:,-1:])
                batch, seq_len = input_ids.shape
                position_ids = repeat(torch.arange(seq_len), 's -> b s', b = batch) 
                positions = self.position_embedding(position_ids[:,-1:])
                embedding = tokens + positions
                x = self.dropout(embedding)
                new_key_values = []
                for i,gptblock in enumerate(self.GPTBlocks):
                    x, new_key_value = gptblock(x, 
                            past_key_values = self.past_key_values[i,:,:,:], 
                            return_key_values = True)
                    new_key_values.append(new_key_value)
                new_key_values = torch.cat(new_key_values, dim = 0)
                self.past_key_values = torch.cat((self.past_key_values, new_key_values), dim=2)
                final_encodings = self.layer_norm(x)[:,-1,:]
                logits = einsum('b c, v c -> b v', final_encodings, self.token_embedding.weight)
                return GPT2Output(logits, final_encodings)
