# A simplified language model based on https://github.com/graykode/gpt-2-Pytorch
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias
    
class Attention(nn.Module):
    def __init__(self, *, n_embd, n_ctx, n_head, dropout):
        super(Attention, self).__init__()
        assert n_embd % n_head == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = n_head
        self.split_size = n_embd
        self.c_attn = nn.Linear(n_embd, n_embd * 3)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def _attn(self, q, k, v):
        w = torch.matmul(q, k)
        # TODO: add trainable temperature here
        w = w / math.sqrt(v.size(-1))
        nd, ns = w.size(-2), w.size(-1)
        b = self.bias[:, :, ns-nd:ns, :ns]
        w = w * b - 1e10 * (1 - b)
        w = F.softmax(w, dim=-1)
        return torch.matmul(w, v)

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
        else:
            return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)

    def forward(self, x, layer_past=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2, -1), layer_past[1]  # transpose back cf below
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
        present = torch.stack((key.transpose(-2, -1), value))  # transpose to have same shapes for stacking
        a = self._attn(query, key, value)
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.dropout(a)
        return a, present

class MLP(nn.Module):
    def __init__(self, n_state, n_embd):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        self.c_fc = nn.Linear(n_embd, n_state)
        self.c_proj = nn.Linear(n_state, n_embd)

    def forward(self, x):
        h = gelu(self.c_fc(x))
        h2 = self.c_proj(h)
        return h2

class Block(nn.Module):
    def __init__(self, *, n_ctx, n_embd, n_state, n_head, dropout):
        super(Block, self).__init__()
        self.ln_1 = LayerNorm(n_embd, eps=1e-8)
        self.attn = Attention(n_embd=n_embd, n_ctx=n_ctx, n_head=n_head, dropout=dropout)
        self.ln_2 = LayerNorm(n_embd, eps=1e-8)
        self.mlp = MLP(4 * n_embd, n_embd)

    def forward(self, x, layer_past=None):
        a, present = self.attn(self.ln_1(x), layer_past=layer_past)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present

class GPT2Model(nn.Module):
    def __init__(self, *, vocab_size, n_positions, n_layer, n_embd, n_ctx, n_state, n_head, dropout, eps=1e-8):
        super(GPT2Model, self).__init__()
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.n_vocab = vocab_size

        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(n_positions, n_embd)
        self.h = nn.ModuleList([Block(n_ctx=n_ctx, n_embd=n_embd, n_state=n_state, n_head=n_head, dropout=dropout) for _ in range(n_layer)])
        self.ln_f = LayerNorm(n_embd, eps=eps)

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights  # Tied weights

    def forward(self, input_ids, past=None):
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
            
        position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long,
                                    device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        presents = []
        for block, layer_past in zip(self.h, past):
            hidden_states, present = block(hidden_states, layer_past)
            presents.append(present)
        hidden_states = self.ln_f(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states.view(*output_shape), presents
    
class GPT2LMHeadModel(nn.Module):
    def __init__(self, **kwargs):
        super(GPT2LMHeadModel, self).__init__()
        self.transformer = GPT2Model(**kwargs)
        
    def forward(self, input_ids, past=None):
        hidden_states, presents = self.transformer(input_ids, past)
        lm_logits = torch.matmul(hidden_states, self.transformer.wte.weight.transpose(-1, -2))
        return lm_logits, presents