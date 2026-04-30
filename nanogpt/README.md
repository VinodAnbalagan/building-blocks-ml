# building-blocks-ml

> Build it to understand it. No magic imports. No black boxes.

This repository is a ground-up reconstruction of the core ideas in modern machine learning — written from scratch, one concept at a time, in clean Python.

The goal is not to reinvent the wheel. The goal is to _understand the wheel so well that you could_.

---

## Projects

### Tier 1 — Foundations

| Project                   | What it teaches                                | Status         |
| ------------------------- | ---------------------------------------------- | -------------- |
| [micrograd](./micrograd/) | Backpropagation & autograd from scratch        | 🔨 In progress |
| [nanoGPT](./nanogpt/)     | Transformers & language modelling from scratch | ⏳ Up next     |
| [minbpe](./minbpe/)       | Byte Pair Encoding tokenization from scratch   | ⏳ Planned     |
| Transformer (NumPy only)  | Attention without any neural net framework     | ⏳ Planned     |

### Tier 2 — Extend & Experiment

| Project                 | What it teaches                               | Status     |
| ----------------------- | --------------------------------------------- | ---------- |
| nanoGPT fine-tuning     | Custom datasets, transfer learning            | ⏳ Planned |
| RAG from scratch        | Vector search + retrieval + generation        | ⏳ Planned |
| LoRA from scratch       | Parameter-efficient fine-tuning               | ⏳ Planned |
| Custom dataset pipeline | Scraping → cleaning → tokenization → training | ⏳ Planned |

### Tier 3 — Portfolio

| Project                              | What it teaches                   | Status     |
| ------------------------------------ | --------------------------------- | ---------- |
| Deployed model on HuggingFace Spaces | Shipping real ML products         | ⏳ Planned |
| Paper reproduction                   | Reading and implementing research | ⏳ Planned |
| Open source contribution             | Working in a real codebase        | ⏳ Planned |

---

## Stack

- Python 3.11+
- PyTorch (used deliberately, not as a crutch)
- marimo (interactive notebooks where it adds value)
- HuggingFace Hub (for deploying models and Spaces)

---

## Writing

Each project has a companion post on [The Meta Gradient](https://substack.com/@themetagradient) — where the ideas behind the code get explained in plain English.

---

## Roadmap

The long arc of this repo points toward original research. The immediate destination is a deep enough understanding of transformers, state space models, and geometric deep learning to implement and contribute to ideas at the frontier.

---

_Built in public. Updated as I go._
