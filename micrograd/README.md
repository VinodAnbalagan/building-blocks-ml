# micrograd

A scalar-valued autograd engine built from scratch, following Karpathy's micrograd.

## What it is

A minimal implementation of reverse-mode automatic differentiation.
The core idea: every operation tracks how to compute its own gradient,
and a topological sort of the computation graph lets us run backprop
automatically on any expression built from `Value` objects.

## What's inside

- `micrograd.py` — the `Value` class, the entire autograd engine
- `nn.py` — `Neuron`, `Layer`, `MLP` built on top of `Value`
- `train.py` — a simple training loop demonstrating a 4-example classification task

## How to run

```bash
uv venv
source .venv/bin/activate
uv add torch
python micrograd/train.py
```

## What I learned

- How backpropagation works at the scalar level
- How the chain rule maps to code via closures
- How a neural network is just repeated application of `+`, `*`, and `tanh`
- Why `optimizer.zero_grad()` exists in PyTorch
