#!/bin/bash

ROOT=$(cd $(dirname $0) && pwd)/..

TMPDIR=/tmp/train_ngram.$$

if [ $# -eq 3 ]; then
    WORKDIR=$3
elif [ $# -eq 2 ]; then
    WORKDIR=$TMPDIR
else
    echo "usage: $0 <infile> <outfile> [<tmpdir>]"
    exit 1
fi

INFILE=$1
OUTFILE=$2
PREFIX=$(basename $OUTFILE)

EPOCHS=10
VOCAB_SIZE=5000
NGRAM_SIZE=3

mkdir -p $WORKDIR

$ROOT/src/prepareNeuralLM --train_text $INFILE --ngram_size $NGRAM_SIZE --vocab_size $VOCAB_SIZE --validation_size 500 --write_words_file $WORKDIR/words --train_file $WORKDIR/train.ngrams --validation_file $WORKDIR/validation.ngrams || exit 1

$ROOT/src/trainNeuralNetwork --train_file $WORKDIR/train.ngrams --validation_file $WORKDIR/validation.ngrams --num_epochs $EPOCHS --words_file $WORKDIR/words --model_prefix $WORKDIR/$PREFIX --learning_rate 1 --minibatch_size 8 || exit 1

cp $WORKDIR/$PREFIX.$(($EPOCHS)) $OUTFILE || exit 1

$ROOT/src/testNeuralNetwork --test_file $WORKDIR/train.ngrams --model_file $OUTFILE || exit 1

$ROOT/src/testNeuralLM --test_file $WORKDIR/train.ngrams --model_file $OUTFILE --numberize 0 --ngramize 0 --add_start_stop 0 > $WORKDIR/train.ngrams.scores || exit 1

# $ROOT/src/testNeuralLM --test_file $INFILE --model_file $OUTFILE --numberize 1 --ngramize 1 --add_start_stop 1 > $WORKDIR/inferno.testNeuralLM.scores || exit 1

rm -rf $TMPDIR
