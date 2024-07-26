import torch
import tiktoken

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

enc = tiktoken.get_encoding("gpt2")
seed = 1337
torch.manual_seed(seed) 
data = torch.tensor(enc.encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


def get_batch(batch_size, context_length, split='train'):
    data = train_data if split == 'train' else val_data
    index = torch.randint(len(data) - context_length, (batch_size,))
    x = torch.stack([data[i:i+context_length] for i in index])
    y = torch.stack([data[i+1:i+1+context_length] for i in index])
    return x, y