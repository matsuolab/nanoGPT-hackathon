class Ngram:
    def __init__(self, n, vocab, laplace=1):
        self.n = n
        self.ngram = {}

        def generate_ngrams(prefix, n):
            if n == 0:
                self.ngram[tuple(prefix)] = laplace
                return
            for word in vocab:
                generate_ngrams(prefix + [word], n - 1)
        generate_ngrams([], self.n)
    
    def train(self, token_list):
        for i in range(len(token_list) - self.n + 1):
            ngram = tuple(token_list[i:i+self.n])
            if ngram in self.ngram:
                self.ngram[ngram] += 1
            else:
                self.ngram[ngram] = 1


if __name__ == "__main__":
    vocab = ["I", "am", "an", "NLPer", "a", "student", "in", "Tokyo", "University"]
    text = "I am an NLPer"
    words = text.split()
    ngram = Ngram(2, vocab)
    ngram.train(words)
    print(ngram.ngram)