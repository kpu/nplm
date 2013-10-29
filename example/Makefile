all: inferno.nnlm

%.nnlm: %.txt
	./train_ngram.sh $^ $@ work

inferno.txt: pg8800.txt
	sed -n 10339,15586p $^ | ./preprocess.pl > $@

purgatorio.txt: pg8800.txt
	sed -n 5135,10307p $^ | ./preprocess.pl > $@

paradiso.txt: pg8800.txt
	sed -n 107,5104p $^ | ./preprocess.pl > $@
