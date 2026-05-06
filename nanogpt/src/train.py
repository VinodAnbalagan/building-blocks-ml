import torch
from data import load_data, build_vocab, encode, get_splits, get_batch
from model import BigramLanguageModel

# hyperparameters
batch_size = 4
block_size = 8
learning_rate = 1e-3
max_steps = 10000
eval_interval = 1000

torch.manual_seed(1337)

# data
text = load_data()
vocab_size, stoi, itos = build_vocab(text)
data = torch.tensor(encode(text, stoi), dtype=torch.long)
train_data, val_data = get_splits(data)

# model
model = BigramLanguageModel(vocab_size)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# training loop
for step in range(max_steps):
    xb, yb = get_batch('train', train_data, val_data, block_size, batch_size)
    logits, loss = model(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % eval_interval == 0:
        print(f'step {step}: loss {loss.item():.4f}')

# generate
print('\n--- Generated text ---')
context = torch.zeros((1, 1), dtype=torch.long)
generated = model.generate(context, max_new_tokens=300)
print(''.join([itos[i.item()] for i in generated[0]]))