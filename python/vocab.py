import heapq
import operator
import sys

class Vocab(object):
    def __init__(self, words=(), unk="<unk>"):
        self.words = []
        self.word_index = {}

        self.insert_word(unk)
        self.unk = self.word_index[unk]
        for word in words:
            self.insert_word(word)

    def from_counts(self, counts, size, unk="<unk>"):
        # Keep only most frequent words
        q = [(-count, word) for (word, count) in counts.iteritems()]
        heapq.heapify(q)
        inserted = 0
        while len(self.words) < size and len(q) > 0:
            _, word = heapq.heappop(q)
            inserted += 1
            if word not in self.word_index:
                self.insert_word(word)
        return inserted

    def insert_word(self, word):
        i = len(self.words)
        self.words.append(word)
        self.word_index[word] = i

    def lookup_word(self, word):
        return self.word_index.get(word, self.unk)
