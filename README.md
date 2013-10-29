2013-07-30

Prerequisites
-------------

Before compiling, you must have the following:

A C++ compiler and GNU make

Boost 1.47.0 or later
http://www.boost.org

Eigen 3.1.x
http://eigen.tuxfamily.org

Optional:

Intel MKL 11.x
http://software.intel.com/en-us/intel-mkl
Recommended for better performance.

Python 2.7.x, not 3.x
http://python.org

Cython 0.19.x
http://cython.org
Needed only for building Python bindings.

Building
--------

To compile, edit the Makefile to reflect the locations of the Boost
and Eigen include directories.

If you want to use the Intel MKL library (recommended if you have it),
uncomment the line
    MKL=/path/to/mkl
editing it to point to the MKL root directory.

By default, multithreading using OpenMP is enabled. To turn it off,
comment out the line
    OMP=1

Then run 'make install'. This creates several programs in the bin/
directory and a library lib/neuralLM.a.

Notes on particular configurations:

- Intel C++ compiler and OpenMP. With version 12, you may get a
  "pragma not found" error. This is reportedly fixed in ComposerXE
  update 9.

- Mac OS X and OpenMP. The Clang compiler (/usr/bin/c++) doesn't
  support OpenMP. If the g++ that comes with XCode doesn't work
  either, try the one installed by MacPorts (/opt/local/bin/g++ or
  /opt/local/bin/g++-mp-*).

Training a language model
-------------------------

Building a language model requires some preprocessing. In addition to
any preprocessing of your own (tokenization, lowercasing, mapping of
digits, etc.), prepareNeuralLM (run with --help for options) does the
following:

- Splits into training and validation data. The training data is used
  to actually train the model, while the validation data is used to
  check its performance.
- Creates a vocabulary of the k most frequent words, mapping all other
  words to <unk>.
- Adds start <s> and stop </s> symbols.
- Converts to numberized n-grams.

A typical invocation would be:

    prepareNeuralLM --train_text mydata.txt --ngram_size 3 \
                    --n_vocab 5000 --words_file words \
                    --train_file train.ngrams \
                    --validation_size 500 --validation_file validation.ngrams

which would generate the files train.ngrams, validation.ngrams, and words.

These files are fed into trainNeuralNetwork (run with --help for
options). A typical invocation would be:

    trainNeuralNetwork --train_file train.ngrams \
                       --validation_file validation.ngrams \
                       --num_epochs 10 \
                       --words_file words \
                       --model_prefix model

After each pass through the data, the trainer will print the
log-likelihood of both the training data and validation data (higher
is better) and generate a series of model files called model.1,
model.2, and so on. You choose which model you want based on the
validation log-likelihood.

You can find a working example in the example/ directory. The Makefile
there generates a language model from a raw text file.

Notes:

- Vocabulary. You should set --n_vocab to something less than the
  actual vocabulary size of the training data (and will receive a
  warning if it's not). Otherwise, no probability will be learned for
  unknown words. On the other hand, there is no need to limit n_vocab
  for the sake of speed. At present, we have tested it up to 100000.

- Normalization. Most of the computational cost normally (no pun
  intended) associated with a large vocabulary has to do with
  normalization of the conditional probability distribution P(word |
  context). The trainer uses noise-contrastive estimation to avoid
  this cost during training (Gutmann and Hyvärinen, 2010), and, by
  default, sets the normalization factors to one to avoid this cost
  during testing (Mnih and Hinton, 2009).

  If you set --normalization 1, the trainer will try to learn the
  normalization factors, and you should accordingly turn on
  normalization when using the resulting model. The default initial
  value --normalization_init 0 should be fine; you can try setting it
  a little higher, but not lower.

- Validation. The trainer computes the log-likelihood of a validation
  data set (which should be disjoint from the training data). You use
  this to decide when to stop training, and the trainer also uses it
  to throttle the learning rate. This computation always uses exact
  normalization and is therefore much slower, per instance, than
  training. Therefore, you should make the validation data
  (--validation_size) as small as you can. (For example, Section 00 of
  the Penn Treebank has about 2000 sentences and 50,000 words.)

Python code
-----------

prepareNeuralLM.py performs the same function as prepareNeuralLM, but in
Python. This may be handy if you want to make modifications.

nplm.py is a pure Python module for reading and using language models
created by trainNeuralNetwork. See testNeuralLM.py for example usage.

In src/python are Python bindings (using Cython) for the C++ code. To
build them, run 'make python/nplm.so'.

Using in a decoder
------------------

To use the language model in a decoder, include neuralLM.h and link
against neuralLM.a. This provides a class nplm::neuralLM, with the
following methods:

    void set_normalization(bool normalization);

Turn normalization on or off (default: off). If normalization is off,
the probabilities output by the model will not be normalized. In
general, this means that summing over all possible words will not give
a probability of one. If normalization is on, computes exact
probabilities (too slow to be recommended for decoding).

    void set_map_digits(char c);

Map all digits (0-9) to the specified character. This should match
whatever mapping you used during preprocessing.

    void set_log_base(double base);

Set the base of the log-probabilities returned by lookup_ngram. The
default is e (natural log), whereas most other language modeling
toolkits use base 10.

    void read(const string &filename);

Read model from file.

    int get_order();

Return the order of the language model.

    int lookup_word(const string &word);

Map a word to an index for use with lookup_ngram().

    double lookup_ngram(const vector<int> &ngram);
    double lookup_ngram(const int *ngram, int n);

Look up the log-probability of ngram.

