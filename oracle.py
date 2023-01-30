class Dummy_oracle():
    def __init__(self, corpus):
        self.corpus = corpus.split(". ")
    def consult(self, query):
        # return the first sentence of the corpus
        return self.corpus[0]
