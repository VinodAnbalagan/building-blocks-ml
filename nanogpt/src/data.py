import urllib.request
import torch 

def download_shakespeare(): 
    url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    urllib.request.urlretrieve(url, 'nanogpt/data/input.txt')


def load_data():
    with open('nanogpt/data/input.txt', 'r') as f: 
        text = f.read() 
    return text 

def build_vocab(text): 
    chars = sorted(list(set(text)))
    vocab_size = len(chars) 
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return vocab_size, stoi, itos 

def encode(text, stoi):
    return [stoi[c] for c in text]

def decode(indices, itos):
    return ''.join([itos[i] for i in indices]) 

def get_splits(data, split_ratio=0.9): 
    n = int(split_ratio * len(data)) 
    train_data = data[:n] 
    val_data = data[n:] 
    return train_data, val_data 

def get_batch(split, train_data, val_data, block_size=8, batch_size=4): 
    data = train_data if split == 'train' else val_data 
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y                   

if __name__ == '__main__': 
    download_shakespeare()
    text = load_data()
    print(f'Total characters: {len(text):,}')

    vocab_size, stoi, itos = build_vocab(text)
    print(f'Vocab size: {vocab_size}')
    print(f'All characters: {"".join(sorted(stoi.keys()))}')

    sample = 'Hello'
    encoded = encode(sample, stoi)
    decoded = decode(encoded, itos)
    print(f'Encoded: {encoded}')
    print(f'Decoded: {decoded}')

    data = torch.tensor(encode(text, stoi), dtype=torch.long)
    train_data, val_data = get_splits(data)
    xb, yb = get_batch('train', train_data, val_data)
    print('Input shape:', xb.shape)
    print('Target shape:', yb.shape)
    print('First input:', xb[0])
    print('First target:', yb[0])
  


    