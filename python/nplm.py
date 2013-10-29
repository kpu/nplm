import numpy
import numpy.random
import scipy.sparse

def diag_dot(a, b, out=None):
    """Input:  a and b are arrays.
       Output: a column vector of the dot product of the rows of a and respective 
       columns of b, in other words, diag(a.dot(b))."""
    if out is None:
        out = numpy.empty((a.shape[0], 1))
    numpy.einsum('ji,ij->j', a, b, out=out[:,0])
    return out

class NeuralLM(object):
    def __init__(self, ngram_size, n_vocab, input_embedding_dimension, n_hidden, output_embedding_dimension):
        self.n_vocab = n_vocab
        self.index_to_word = []
        self.word_to_index = {}

        self.ngram_size = ngram_size
        self.input_embedding_dimension = input_embedding_dimension
        self.n_hidden = n_hidden
        self.output_embedding_dimension = output_embedding_dimension

        self.input_embeddings = numpy.zeros((n_vocab,             input_embedding_dimension))
        self.hidden1_weights  = numpy.zeros((n_hidden,            (ngram_size-1)*input_embedding_dimension))
        self.hidden2_weights  = numpy.zeros((output_embedding_dimension, n_hidden))
        self.output_weights   = numpy.zeros((n_vocab,             output_embedding_dimension))
        self.output_biases    = numpy.zeros((n_vocab,             1))

    def initialize(self, r):
        def uniform(m):
            m[:,:] = numpy.random.uniform(-r, r, m.shape)
        uniform(self.input_embeddings)
        uniform(self.hidden1_weights)
        uniform(self.hidden2_weights)
        uniform(self.output_weights)
        uniform(self.output_biases)

    def forward_prop(self, inputs, output=None, normalize=True):
        u = numpy.bmat([[self.input_embeddings.T * ui] for ui in inputs])
        h1 = numpy.maximum(0., self.hidden1_weights * u)
        h2 = numpy.maximum(0., self.hidden2_weights * h1)

        if output is None:
            o = self.output_weights * h2 + self.output_biases
        else:
            # Inefficient version:
            #o = diag_dot(output.T, (self.output_weights * h2 + self.output_biases))
            #o = output.multiply(self.output_weights * h2 + self.output_biases)

            # Since output is sparse, distributing multiplication by output
            # is much more efficient:

            o = diag_dot(output.T * self.output_weights, h2) + output.T * self.output_biases
        return o

    def backward_prop(self, g_output):
        pass

    def to_file(self, outfile):

        def write_matrix(m):
            for i in xrange(m.shape[0]):
                outfile.write("\t".join(map(str, m[i])))
                outfile.write("\n")
            outfile.write("\n")

        def write_vector(m):
            for i in xrange(m.shape[0]):
                outfile.write(str(m[i]))
                outfile.write("\n")
            outfile.write("\n")

        outfile.write("\\config\n")
        outfile.write("version 1\n")
        outfile.write("ngram_size %d\n" % self.ngram_size)
        outfile.write("n_vocab %d\n" % self.n_vocab)
        outfile.write("input_embedding_dimension %d\n" % self.input_embedding_dimension)
        outfile.write("output_embedding_dimension %d\n" % self.output_embedding_dimension)
        outfile.write("n_hidden %d\n" % self.n_hidden)
        outfile.write("\n")

        outfile.write("\\vocab\n")
        for word in self.index_to_word:
            outfile.write(word + "\n")
        outfile.write("\n")

        outfile.write("\\input_embeddings\n")
        write_matrix(self.input_embeddings)

        outfile.write("\\hidden_weights 1\n")
        write_matrix(self.hidden1_weights)

        outfile.write("\\hidden_weights 2\n")
        write_matrix(self.hidden2_weights)

        outfile.write("\\output_weights\n")
        write_matrix(self.output_weights)

        outfile.write("\\output_biases\n")
        write_matrix(self.output_biases)

        outfile.write("\\end\n")

    @staticmethod
    def from_file(infile):
        """Create a NeuralLM from a text file."""

        # Helper functions
        def read_sections(infile):
            while True:
                line = infile.next().strip()
                if line == "\\end":
                    break
                elif line.startswith('\\'):
                    yield line, read_section(infile)

        def read_section(infile):
            while True:
                line = infile.next().strip()
                if line == "":
                    break
                else:
                    yield line

        def read_matrix(lines, m, n, out=None):
            if out is None:
                out = numpy.zeros((m, n))
            i = 0
            for line in lines:
                row = numpy.array(map(float, line.split()))
                if len(row) != n:
                    raise Exception("wrong number of columns (expected %d, found %d)" % (n, len(row)))
                if i >= m:
                    raise Exception("wrong number of rows (expected %d, found more)" % m)
                out[i,:] = row
                i += 1
            if i < m:
                raise Exception("wrong number of rows (expected %d, found %d)" % (m, i))
            return out

        if isinstance(infile, str):
            infile = open(infile)

        for section, lines in read_sections(infile):
            if section == "\\config":
                config = {}
                for line in lines:
                    key, value = line.split()
                    config[key] = value

                m = NeuralLM(ngram_size=int(config['ngram_size']),
                             n_vocab=int(config['n_vocab']),
                             input_embedding_dimension=int(config['input_embedding_dimension']),
                             n_hidden=int(config['n_hidden']),
                             output_embedding_dimension=int(config['output_embedding_dimension']))

            elif section == "\\vocab":
                for line in lines:
                    m.index_to_word.append(line)
                m.word_to_index = dict((w,i) for (i,w) in enumerate(m.index_to_word))

            elif section == "\\input_embeddings":
                read_matrix(lines, m.n_vocab, m.input_embedding_dimension, out=m.input_embeddings)
            elif section == "\\hidden_weights 1":
                read_matrix(lines, m.n_hidden, (m.ngram_size-1)*m.input_embedding_dimension, out=m.hidden1_weights)
            elif section == "\\hidden_weights 2":
                read_matrix(lines, m.output_embedding_dimension, m.n_hidden, out=m.hidden2_weights)
            elif section == "\\output_weights":
                read_matrix(lines, m.n_vocab, m.output_embedding_dimension, out=m.output_weights)
            elif section == "\\output_biases":
                read_matrix(lines, m.n_vocab, 1, out=m.output_biases)
        return m

    def make_data(self, ngrams):
        """Takes a list of n-grams of words (as ints),
           and converts into a list of n sparse arrays."""
        rows = [[] for j in xrange(self.ngram_size)]
        cols = [[] for j in xrange(self.ngram_size)]
        values = [[] for j in xrange(self.ngram_size)]
        for i, ngram in enumerate(ngrams):
            for j, w in enumerate(ngram):
                rows[j].append(w)
                cols[j].append(i)
                values[j].append(1)
        data = [scipy.sparse.csc_matrix((values[j], (rows[j], cols[j])), shape=(self.n_vocab, len(ngrams))) for j in xrange(self.ngram_size)]
        return data
