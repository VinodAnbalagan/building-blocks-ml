import torch
import argparse
from model import BigramLanguageModel

def load_model(checkpoint_path='nanogpt/checkpoints/model.pt'):
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    model = BigramLanguageModel(
        vocab_size=checkpoint['vocab_size'],
        n_embd=checkpoint['n_embd'],
        block_size=checkpoint['block_size'],
        n_head=checkpoint['n_head'],
        n_layer=checkpoint['n_layer'],
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint['stoi'], checkpoint['itos']

def generate(prompt, max_new_tokens=500):
    model, stoi, itos = load_model()
    
    # encode prompt
    if prompt == '':
        idx = torch.zeros((1, 1), dtype=torch.long)
    else:
        idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long)
    
    # generate
    output = model.generate(idx, max_new_tokens)
    text = ''.join([itos[i.item()] for i in output[0]])
    
    return text

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, default='', help='Starting prompt')
    parser.add_argument('--tokens', type=int, default=500, help='Tokens to generate')
    args = parser.parse_args()
    
    print(generate(args.prompt, args.tokens))