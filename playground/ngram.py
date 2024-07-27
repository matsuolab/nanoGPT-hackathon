from collections import defaultdict
import torch
from torch.nn import functional as F
import math


class Ngram:
    def __init__(self, n, vocab, laplace=1):
        self.n = n
        self.vocab = vocab
        self.laplace = laplace
        self.ngram = defaultdict(lambda: laplace)
        self.context_count = defaultdict(lambda: laplace * len(self.vocab))
    
    def train(self, token_list):
        for i in range(len(token_list) - self.n + 1):
            ngram = tuple(token_list[i:i+self.n])
            # print(ngram)
            context = ngram[:-1]
            self.ngram[ngram] += 1
            self.context_count[context] += 1
    

    def train_batch(self, token_list):
        for tokens in token_list:
            self.train(tokens)
    
    def get_prob(self, ngram):
        if self.n == 1:
            return self.ngram[ngram] / len(self.vocab)
        else:
            context = ngram[:-1]
            # if self.context_count[context] == 0:
            #     return 1 / len(self.vocab)
            # else:
            #     if self.ngram[ngram] == 0:
            #         return 1e-20
            #     return self.ngram[ngram] / self.context_count[context]
            return self.ngram[ngram] / self.context_count[context]
    
    def get_prob_distribution(self, n_minus_1_gram):
        distribution = []
        distribution_dict = {}
        context = tuple(n_minus_1_gram)
        for word in self.vocab:
            ngram = context + (word,)
            # print('hi', ngram)
            distribution.append(self.get_prob(ngram))
            distribution_dict[word] = self.get_prob(ngram)
        return distribution, distribution_dict
    
    def forward(self, token_indexes):
        # token_index: (batch_size, sequence_length)
        if type(token_indexes) == torch.Tensor:
            token_indexes = token_indexes.tolist()
        batch_size = len(token_indexes)
        sequence_length = len(token_indexes[0])
        distributions = torch.ones(batch_size, sequence_length, len(self.vocab))
        distributions /= len(self.vocab)
        for i in range(sequence_length):
            for batch in range(batch_size):
                if self.n == 2:
                    context = tuple([i])
                else:
                    context = tuple(token_indexes[batch][max(0, i-self.n+2):i+1])
                distribution, _ = self.get_prob_distribution(context)
                distributions[batch, i] = torch.tensor(distribution)
        # distributions: (batch_size, sequence_length, vocab_size)
        return distributions
    
    def loss(self, token_indexes, targets):
        # token_indexes: (batch_size, sequence_length)
        # targets: (batch_size, sequence_length)
        distributions = self.forward(token_indexes)
        distributions = distributions.to(targets.device)
        log_distributions = torch.log(distributions)
        # print(log_distributions)
        # targets: (batch_size, sequence_length)
        batch_size, sequence_length, vocab_size = log_distributions.shape
        loss = F.nll_loss(
            log_distributions.view(batch_size*sequence_length, vocab_size),
            targets.view(batch_size*sequence_length)
            )
        # loss: scalar
        return loss
        


if __name__ == "__main__":
    vocab_str = ["I", "am", "an", "NLPer", "a", "student", "in", "Tokyo", "University"]
    tokenizer = {
        "I": 0,
        "am": 1,
        "an": 2,
        "NLPer": 3,
        "a": 4,
        "student": 5,
        "in": 6,
        "Tokyo": 7,
        "University": 8
    }
    decoder = {v: k for k, v in tokenizer.items()}
    text = "I am an NLPer"
    words = text.split()
    words_token = [tokenizer[word] for word in words]
    ngram = Ngram(2, tokenizer.values())
    # distribution, distribution_dict = ngram.get_prob_distribution((tokenizer["I"],))
    # print(distribution)
    # print(distribution_dict)
    x, y = words[:-1], words[1:]
    x = [[tokenizer[word] for word in x]]
    y = torch.tensor([[tokenizer[word] for word in y]])
    print(x, y)
    loss = ngram.loss(x, y)
    print(loss)

    ngram.train(words_token)
    distribution, distribution_dict = ngram.get_prob_distribution((tokenizer["I"],))
    print(distribution)
    print(distribution_dict)
    loss = ngram.loss(x, y)
    print(loss)

    ngram = Ngram(2, tokenizer.values(), 1e-5)
    ngram.train(words_token)
    loss = ngram.loss(x, y)
    print(loss)

    # ngram = Ngram(2, tokenizer.values())
    # for epoch in range(100):
    #     ngram.train(words_token)
    #     loss = ngram.loss(x, y)
    #     print('Epoch: {}, Loss: {}'.format(epoch, loss))
    # distribution, distribution_dict = ngram.get_prob_distribution((tokenizer["I"],))
    # print(distribution)
    # print(distribution_dict)