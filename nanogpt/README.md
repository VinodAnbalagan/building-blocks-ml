# nanoGPT

A decoder-only transformer (GPT) built from scratch, following Karpathy's nanoGPT. Character-level, trained on tiny Shakespeare.

## What it is

A minimal GPT built one component at a time, so that nothing in the architecture is a black box. It starts from a plain bigram model and adds — in order — embeddings, self-attention, multi-head attention, a feed-forward network, and finally residual connections and LayerNorm, stacked into transformer blocks. The same building blocks, scaled up, are what GPT-2 and GPT-3 are made of.

## What's inside

- `src/data.py` — downloads tiny Shakespeare, builds the character vocab, encode/decode, train/val split, batching
- `src/model.py` — `Head`, `MultiHeadAttention`, `FeedForward`, `Block`, and the full `BigramLanguageModel` (a GPT, despite the name it grew out of)
- `src/train.py` — training loop with train/val loss estimation and checkpointing
- `src/sample.py` — loads a saved checkpoint and generates text from any prompt

## How to run

```bash
# from the repo root, with the venv active
python nanogpt/src/train.py                          # trains and saves a checkpoint
python nanogpt/src/sample.py --prompt "ROMEO:" --tokens 300
```

Training the scaled config (4 layers, 4 heads, 128-dim embeddings, block size 64) reaches a validation loss around 1.7 and produces recognisable Shakespeare-style text — real character names, speaker tags, and mostly-correct grammar.

## What I learned

- How self-attention moves information between positions via query / key / value
- Why causal masking (`-inf` before softmax) keeps the model from seeing the future
- What every tensor shape in a transformer block represents, and why `view(B*T, C)` is needed for the loss
- Why residual connections and LayerNorm are what make deep transformers trainable
- How token and position embeddings combine to give the model both identity and order

## A note on the name

The class is called `BigramLanguageModel` for historical reasons — it began as a literal bigram model and grew into a full transformer as each component was added. The name was kept to preserve that lineage.
