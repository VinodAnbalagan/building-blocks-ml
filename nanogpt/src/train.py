import torch
from data import load_data, build_vocab, encode, get_splits, get_batch
from model import BigramLanguageModel

# hyperparameters
batch_size = 32
block_size = 64
learning_rate = 3e-4
max_steps = 5000
eval_interval = 500
eval_iters = 200
n_embd = 128
n_head = 4 
n_layer = 4 

torch.manual_seed(1337)

# data
text = load_data()
vocab_size, stoi, itos = build_vocab(text)
data = torch.tensor(encode(text, stoi), dtype=torch.long)
train_data, val_data = get_splits(data)

# model
model = BigramLanguageModel(vocab_size, n_embd, block_size, n_head, n_layer)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad() 
def estimate_loss(): 
    out = {} 
    model.eval() 
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split, train_data, val_data, block_size, batch_size)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out 

# training loop
for step in range(max_steps):
    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    xb, yb = get_batch('train', train_data, val_data, block_size, batch_size)
    _, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# generate
print('\n--- Generated text ---')
context = torch.zeros((1, 1), dtype=torch.long)
generated = model.generate(context, max_new_tokens=300)
print(''.join([itos[i.item()] for i in generated[0]]))

# save model
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab_size': vocab_size,
    'n_embd': n_embd,
    'block_size': block_size,
    'n_head': n_head,
    'n_layer': n_layer,
    'stoi': stoi,
    'itos': itos,
}, 'nanogpt/checkpoints/model.pt')
print('\nModel saved to nanogpt/checkpoints/model.pt')