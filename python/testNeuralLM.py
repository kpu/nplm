import nplm

if __name__ == "__main__":
    import sys
    import fileinput
    import argparse

    parser = argparse.ArgumentParser(description='Score sentences using n-gram language model.')
    parser.add_argument('--test_file', metavar='file', dest='test_file', help='test text file')
    parser.add_argument('--model_file', metavar='file', dest='model_file', help='model file')
    args = parser.parse_args()

    m = nplm.NeuralLM.from_file(args.model_file)
    n = m.ngram_size
    for line in fileinput.input(args.test_file):
        words = line.split()
        if len(words) < n: continue
        unk = m.word_to_index['<unk>']
        words = ['<s>'] * (n-1) + words + ['</s>']
        words = [m.word_to_index.get(w, unk) for w in words]
        ngrams = []
        for i in xrange(n-1, len(words)):
            ngrams.append(words[i-n+1:i+1])
        ngrams = m.make_data(ngrams)
        print m.forward_prop(ngrams[:-1], output=ngrams[-1])[:,0]
