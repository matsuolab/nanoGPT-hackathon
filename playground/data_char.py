import torch

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

class CharTokenizer:
    def __init__(self, text):
        self.vocab = sorted(list(set(text)))
        self.n_vocab = len(self.vocab)
        self.encoder = {k: v for v, k in enumerate(self.vocab)}
        self.decoder = {v: k for k, v in self.encoder.items()}
    
    def encode(self, text):
        return [self.encoder[c] for c in text]
    
    def decode(self, tokens):
        return [self.decoder[t] for t in tokens]

seed = 1337
torch.manual_seed(seed)
enc = CharTokenizer(text)
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