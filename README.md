# building-blocks-ml

> Build it to understand it. No magic imports. No black boxes.

Two projects, built from scratch in clean Python, each rebuilding a foundational idea in modern machine learning line by line. The goal was never to reinvent the wheel — it was to understand the wheel so well that I could.

Both are complete.

---

## Projects

### [micrograd](./micrograd/) — backpropagation from scratch

A scalar-valued autograd engine. A `Value` class where every operation records how to compute its own gradient, and a topological sort runs backprop automatically over any expression graph. A small `Neuron` / `Layer` / `MLP` stack is built on top and trained with a hand-written loop.

**What it teaches:** what `loss.backward()` is actually doing, how the chain rule maps to code via closures, and why `optimizer.zero_grad()` has to exist.

### [nanoGPT](./nanogpt/) — a GPT from scratch

A decoder-only transformer built up one component at a time — token and position embeddings, single-head self-attention, multi-head attention, feed-forward layers, residual connections and LayerNorm — then stacked into blocks and trained at the character level on tiny Shakespeare. Includes checkpointing and an inference script that generates from any prompt.

**What it teaches:** how attention actually moves information between positions, why causal masking matters, what each shape in a transformer is doing, and how the same building blocks scale from this to GPT-2 and beyond.

---

## The arc

The two projects connect directly: micrograd builds the autograd machinery that PyTorch gives you for free, and nanoGPT then leans on that same machinery (via PyTorch) to build something real. Backprop first, then a language model that depends on it — the full path from the chain rule to generated text.

---

## Stack

- Python 3.11+ (managed with uv)
- PyTorch — used deliberately, understood rather than assumed

---

## Writing

Companion posts on [The Meta Gradient](https://substack.com/@themetagradient) explain the ideas behind the code in plain English.

---

*Built in public, to understand — not to impress.*
