import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embed = 32 # size of embedding vector
# ------------

torch.manual_seed(1337)

# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars)) #all the individual characters present in the dataset
print(vocab_size)

stoi = { ch:i for i,ch in enumerate(chars) } #iterates through the characters to create a lookup table
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

print(encode("hii there"))
print(decode(encode("hii there")))

#sentencepiece encodes sub words (not individual characters or words)
data = torch.tensor(encode(text), dtype=torch.long) #encoding the dataset and store it in a tensor
print(data.shape, data.dtype)
print(data[:1000])

#splitting the data
n= int(0.9*len(data)) #90% data will be used to train rest to validate
train_data = data[:n]
val_data = data[n:]

# block_size = 8 #max len for a chunk that is sent ot a transformer at once
# train_data[:block_size+1]

# x = train_data[:block_size]
# y = train_data[1:block_size+1]
# for t in range(block_size):
#   context = x[:t+1]
#   target = y[t]
#   print(f"input : {context}, target : {target}") #Training the model on all the different levels of context to allow the transformer to predict the next character

# torch.manual_seed(1337)
# batch_size = 4 # how many independent sequences will we process in parallel?
# block_size = 8 # what is the maximum context length for predictions?

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,)) #random positions to grab chunks
    x = torch.stack([data[i:i+block_size] for i in ix]) #stacks the 1d tensors and stack them in a 4 by 8 tensor
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad() #function tells pytorch we will not be calling backward here
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# xb, yb = get_batch('train')
# print('inputs:')
# print(xb.shape)
# print(xb)
# print('targets:')
# print(yb.shape)
# print(yb)

print('----')

# for b in range(batch_size): # batch dimension
#     for t in range(block_size): # time dimension
#         context = xb[b, :t+1]
#         target = yb[b,t]
#         print(f"when input is {context.tolist()} the target: {target}")

#transformer is going to look at inputs understand the corresponding target integers present in tensor y



class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed) #token embeddig table
        self.lm_head = nn.Linear(n_embed, vocab_size) #linear layer to project the embedding to the vocab size

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        tok_embeddings = self.token_embedding_table(idx) # (Batch by Time by Channel) cordinates
        logits = self.lm_head(tok_embeddings) # (B, T, vocab_size) logits for next token

        if targets == None:
          loss = None

        else:
          B, T, C = logits.shape
          logits = logits.view(B*T, C) #converting it to 2d for pytorch
          targets = targets.view(B*T)
          loss = F.cross_entropy(logits, targets) #loss fucnrtion to measure the performance of a model

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)
# logits, loss = m(xb,yb)
# print(logits.shape)
# print(loss)
# print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate) #learning rate #pytorch optimiser

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

#The context was based on the last character now the tokens have to talk to each other to understand context