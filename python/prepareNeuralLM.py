#!/usr/bin/env python

import vocab
import collections

start = "<s>"
stop = "</s>"
null = "<null>"

def ngrams(words, n):
    for i in xrange(n-1, len(words)):
        yield words[i-n+1:i+1]

if __name__ == "__main__":
    import sys
    import fileinput
    import argparse

    parser = argparse.ArgumentParser(description='Preprocess training data for n-gram language model.')
    parser.add_argument('--train_text', metavar='file', dest='train_text', help='training text file')
    parser.add_argument('--validation_text', metavar='file', dest='validation_text', help='validation text file (overrides --validation_size)')
    parser.add_argument('--ngram_size', metavar='n', dest='ngram_size', type=int, default=3, help='size of n-grams')
    parser.add_argument('--n_vocab', metavar='V', dest='n_vocab', type=int, help='number of word types')
    parser.add_argument('--words_file', metavar='file', dest='words_file', help='make vocabulary')
    parser.add_argument('--train_file', metavar='file', dest='train_file', default='-', help='make training file')
    parser.add_argument('--validation_file', metavar='file', dest='validation_file', help='make training file')
    parser.add_argument('--validation_size', metavar='m', dest='validation_size', type=int, default=0, help="select m lines for validation")
    args = parser.parse_args()

    n = args.ngram_size
    train_data = []
    validation_data = []
    
    for li, line in enumerate(file(args.train_text)):
        words = line.split()
        words = [start] * (n-1) + words + [stop]
        train_data.append(words)

    if args.validation_text:
        for li, line in enumerate(file(args.validation_text)):
            words = line.split()
            words = [start] * (n-1) + words + [stop]
            validation_data.append(words)
    else:
        if args.validation_size > 0:
            validation_data = train_data[-args.validation_size:]
            train_data[-args.validation_size:] = []

    c = collections.Counter()
    for words in train_data:
        c.update(words[n-1:])

    v = vocab.Vocab()
    v.insert_word(start)
    v.insert_word(stop)
    v.insert_word(null)
    inserted = v.from_counts(c, args.n_vocab)
    if inserted == len(c):
        sys.stderr.write("warning: only %d words types in training data; set --n_vocab lower to learn unknown word\n");

    if args.words_file:
        with open(args.words_file, "w") as outfile:
            for w in v.words:
                outfile.write("%s\n" % (w,))

    if args.train_file == '-':
        outfile = sys.stdout
    else:
        outfile = open(args.train_file, 'w')
    for words in train_data:
        for ngram in ngrams(words, n):
            outfile.write(" ".join(str(v.lookup_word(w)) for w in ngram) + "\n")
    if outfile is not sys.stdout:
        outfile.close()

    if args.validation_file:
        with open(args.validation_file, 'w') as outfile:
            for words in validation_data:
                for ngram in ngrams(words, n):
                    outfile.write(" ".join(str(v.lookup_word(w)) for w in ngram) + "\n")
