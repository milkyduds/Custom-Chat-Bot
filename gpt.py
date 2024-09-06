import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy
import spacy
import pandas as pd
from textblob import TextBlob

#hyperparameters
batch_size = 64 # num of sequences being processed simultaneously in parallel
context_length = 256 #how far behind the sequence should look
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu' # run on gpu if available (faster)
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

class Head(nn.Module):
    '''one head of self-attention'''

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        B,T,C = x.shape
        k = self.key(x) # B x T x C
        q = self.query(x) # B x T x C
        #computer attention scores("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 #(B x T x C) @ (B x C x T) -> (B x T x T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # B x T x T
        wei = F.softmax(wei, dim = -1) # B x T x T
        wei = self.dropout(wei)
        #perform weighted aggregation of values
        v = self.value(x) # B x T x C
        out = wei @ v #(BxTxT) @ (BxTxC) -> (BxTxC)
        return out

class MultiHeadAttention(nn.Module):
    '''multiple heads of self-attention in parallel'''
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    '''simple linear layer followed by non-linearity'''
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    '''transformer block: communication followed by computation'''
    def __init__(self, n_embd, n_head):
        #n_embd: embedding dim, n_head: number of heads
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size) #MHSA, 4 heads of 8-dimensional self-attention
        self.ffwd = FeedForward(n_embd) #Adds linearity
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x): # x is data getting passed through. total_emb
        x = x + self.sa(self.ln1(x)) # apply one head of self attention (B x T xC)
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(context_length, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head = n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size) #language model head

    def forward(self, idx, targets = None):
        B,T = idx.shape
        #idx and targets are both B x T tensors of ints
        tok_emb = self.token_embedding_table(idx) # B x T x n_embd
        pos_emb = self.position_embedding_table(torch.arange(T, device = device)) # T x C
        total_emb = tok_emb + pos_emb # B x T x C
        total_emb = self.blocks(total_emb) # B x T x C
        logits = self.lm_head(total_emb) #B x T x Vocab_size

        if targets is None:
            loss = None
        else:
            B,T,V = logits.shape
            logits = logits.view(B*T, V)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) #identify cost/loss
        return logits, loss

    def generate(self, idx, max_new_tokens):
        #idx is B x T array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_length:] #crop idx to lsat context_length tokens
            logits,loss = self(idx_cond) #get predictions
            logits = logits[:,-1,:] #focus on last time step, B x V
            probabilities = F.softmax(logits, dim = -1) # B x V
            idx_next = torch.multinomial(probabilities, num_samples = 1) # B x 1
            idx = torch.cat((idx, idx_next), dim = 1) # (B, T+1)
        return idx
    
df = pd.read_csv('newsletter_data.csv')
total_files = len(df['text'])
text = "".join(df['text'][0:total_files])

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = { ch:i for i,ch in enumerate(chars)}
itos = { i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encoder: take string, output list of ints
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype = torch.long)

n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i:i+context_length] for i in ix])
    y = torch.stack([data[i+1:i+context_length+1] for i in ix])
    x,y = x.to(device), y.to(device)
    return x,y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits,loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for iter in range(max_iters):
    
    #every once in awhile evalute the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    #sample a batch of data
    xb, yb = get_batch('train')

    #evaluate the loss
    logits,loss = model(xb,yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward()
    optimizer.step()

#generate from the model
context = torch.zeros((1,1), dtype = torch.long, device = device)
print(decode(m.generate(idx = torch.zeros((1,1), dtype = torch.long), max_new_tokens=500)[0].tolist()))