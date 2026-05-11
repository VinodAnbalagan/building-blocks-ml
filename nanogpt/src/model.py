import torch 
import torch.nn as nn 
from torch.nn import functional as F 

class Head(nn.Module):
    
    def __init__(self, head_size, n_embd, block_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.head_size = head_size

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)
        
        # attention scores
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        
        # weighted aggregation
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v      # (B, T, head_size)
        return out

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size, n_embd, block_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_heads = MultiHeadAttention(4, n_embd//4, n_embd, block_size)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)     # (B, T, n_embd)
        pos_emb = self.position_embedding_table(
            torch.arange(T)
        )                                              # (T, n_embd)
        x = tok_emb + pos_emb                         # (B, T, n_embd)
        x = self.sa_heads(x)                           # (B, T, n_embd)
        logits = self.lm_head(x)                      # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

class MultiHeadAttention(nn.Module): 

    def __init__(self, num_heads, head_size, n_embd, block_size): 
        super().__init__()   
        self.heads = nn.ModuleList(
            [Head(head_size, n_embd, block_size) for _ in range(num_heads)] 
        )     
        self.proj = nn.Linear(n_embd, n_embd) 

    def forward(self, x): 
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out) 
        return out 




if __name__ == '__main__':
    B, T, C = 4, 8, 32
    head_size = 16
    
    x = torch.randn(B, T, C)
    head = Head(head_size, n_embd=C, block_size=T)
    out = head(x)
    print('Input shape:', x.shape)
    print('Output shape:', out.shape)