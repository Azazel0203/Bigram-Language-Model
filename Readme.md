# Bigram Language Model

This code implements a simple Bigram Language Model using PyTorch. The Bigram model is a basic language model that predicts the next token based on the previous token. The README provides an overview of the code structure, model architecture, hyperparameters, data processing, training, and generation.

## Model Architecture

### Embedding Layer

The model consists of a token embedding layer:

```python
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
```

## Hyperparameters

```python
# hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# ... (rest of the code)
```

## Data Processing

The training data is loaded from a text file ('input.txt'). Characters are tokenized and mapped to integers for model input.

```python
# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
```

## Training

```python
# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # ... (previous code)
    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

## Generation

```python
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
```

## Running the Code

1. Install the required dependencies, including PyTorch.
2. Download the training data (e.g., a text file) and save it as 'input.txt'.
3. Run the provided code to train the Bigram language model.

## Results

- The model's performance is evaluated by periodically calculating the loss on both the training and validation sets.
- After training, the model generates new text based on a given initial context.