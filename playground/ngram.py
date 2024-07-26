from collections import defaultdict

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
            context = ngram[:-1]
            self.ngram[ngram] += 1
            self.context_count[context] += 1
    
    def get_prob(self, ngram):
        if self.n == 1:
            return self.ngram[ngram] / len(self.vocab)
        else:
            context = ngram[:-1]
            return self.ngram[ngram] / self.context_count[context]
    
    def get_prob_distribution(self, n_minus_1_gram):
        distribution = {}
        context = tuple(n_minus_1_gram)
        for word in self.vocab:
            ngram = context + (word,)
            distribution[word] = self.get_prob(ngram)
        return distribution


if __name__ == "__main__":
    vocab = ["I", "am", "an", "NLPer", "a", "student", "in", "Tokyo", "University"]
    text = "I am an NLPer"
    words = text.split()
    ngram = Ngram(2, vocab)
    ngram.train(words)
    print(dict(ngram.ngram))
    print(ngram.get_prob(("I", "am")))
    print(ngram.get_prob_distribution(("I",)))
