
# GPT From Scratch

A minimal implementation of a **Generative Pretrained Transformer (GPT)** built from scratch in PyTorch, based on the seminal paper _["Attention is All You Need"](https://arxiv.org/abs/1706.03762)_ and inspired by [Andrej Karpathy's](https://github.com/karpathy) educational lecture series.

This project trains a character-level Transformer model on a tiny version of Shakespeare's text to predict and generate new text in a GPT-style fashion.

---

## Features

- Character-level language modeling
- Custom multi-head self-attention from scratch
- Transformer blocks with residual connections & layer normalization
- Text generation with sampling
- Clean and minimal codebase (~300 lines)
- Based on core concepts from the GPT family of models

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/gpt-from-scratch.git
cd gpt-from-scratch
```

### 2. Download the dataset

```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

### 3. Install requirements

```bash
pip install torch
```

### 4. Train the model

```bash
python main.py
```

---

## How It Works

### Tokenization & Encoding

- Reads the dataset (`input.txt`) and builds a character-level vocabulary.
- Each character is assigned a unique integer ID.
- The entire text is encoded as a tensor of integers.

### Model Architecture

- **Token Embeddings**: Maps input character IDs to dense vectors.
- **Positional Embeddings**: Injects token position information.
- **Transformer Blocks**: Multiple stacked blocks with:
  - Multi-Head Self-Attention
  - Feedforward layers
  - Residual connections & LayerNorm
- **Language Modeling Head**: Outputs logits over vocabulary to predict the next character.

### Training

- Trains on batches of sequences using cross-entropy loss.
- Periodically evaluates on a validation set.
- After training, the model can generate Shakespeare-like text.

---

## Acknowledgments

This project is heavily inspired by:

- [Andrej Karpathy’s GPT tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- ["Attention is All You Need" paper](https://arxiv.org/abs/1706.03762)

---

## License

MIT License
