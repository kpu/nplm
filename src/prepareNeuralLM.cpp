#include <iostream>
#include <vector>
#include <queue>
#include <boost/unordered_map.hpp>
#include <tclap/CmdLine.h>
#include <boost/algorithm/string/join.hpp>

#include "neuralLM.h"
#include "util.h"

using namespace std;
using namespace TCLAP;
using namespace boost;
using namespace nplm;

void writeNgrams(const vector<vector<string> > &data, 
		 int ngram_size, const vocabulary &vocab, 
		 bool numberize, bool add_start_stop, bool ngramize, 
		 const string &filename)
{
    ofstream file(filename.c_str());
    if (!file)
    {
	cerr << "error: could not open " << filename << endl;
	exit(1);
    }

    vector<vector<int> > ngrams;
    for (int i=0; i<data.size(); i++) {
        preprocessWords(data[i], ngrams, ngram_size, vocab, numberize, add_start_stop, ngramize);
	// write out n-grams
	for (int j=0; j<ngrams.size(); j++)
	  {
	    for (int k=0; k<ngram_size; k++)
	      {
	        file << ngrams[j][k] << " ";
	      }
	    file << endl;
	  }
    }
    file.close();
}

int main(int argc, char *argv[])
{
    int ngram_size, vocab_size, validation_size;
    bool numberize, ngramize, add_start_stop;
    string train_text, train_file, validation_text, validation_file, words_file, write_words_file;

    try
    {
	CmdLine cmd("Prepares training data for training a language model.", ' ', "0.1");

	// The options are printed in reverse order

    ValueArg<bool> arg_ngramize("", "ngramize", "If true, convert lines to ngrams. Default: true.", false, true, "bool", cmd);
    ValueArg<bool> arg_numberize("", "numberize", "If true, convert words to numbers. Default: true.", false, true, "bool", cmd);
    ValueArg<bool> arg_add_start_stop("", "add_start_stop", "If true, prepend <s> and append </s>. Default: true.", false, true, "bool", cmd);

    ValueArg<int> arg_vocab_size("", "vocab_size", "Vocabulary size.", false, -1, "int", cmd);
    ValueArg<string> arg_words_file("", "words_file", "File specifying words that should be included in vocabulary; all other words will be replaced by <unk>.", false, "", "string", cmd);
    ValueArg<int> arg_ngram_size("", "ngram_size", "Size of n-grams.", true, -1, "int", cmd);
	ValueArg<string> arg_write_words_file("", "write_words_file", "Output vocabulary.", false, "", "string", cmd);
    ValueArg<int> arg_validation_size("", "validation_size", "How many lines from training data to hold out for validation. Default: 0.", false, 0, "int", cmd);
	ValueArg<string> arg_validation_file("", "validation_file", "Output validation data (numberized n-grams).", false, "", "string", cmd);
	ValueArg<string> arg_validation_text("", "validation_text", "Input validation data (tokenized). Overrides --validation_size. Default: none.", false, "", "string", cmd);
	ValueArg<string> arg_train_file("", "train_file", "Output training data (numberized n-grams).", false, "", "string", cmd);
	ValueArg<string> arg_train_text("", "train_text", "Input training data (tokenized).", true, "", "string", cmd);

	cmd.parse(argc, argv);

	train_text = arg_train_text.getValue();
	train_file = arg_train_file.getValue();
	validation_text = arg_validation_text.getValue();
	validation_file = arg_validation_file.getValue();
	validation_size = arg_validation_size.getValue();
	write_words_file = arg_write_words_file.getValue();
	ngram_size = arg_ngram_size.getValue();
	vocab_size = arg_vocab_size.getValue();
	words_file = arg_words_file.getValue();
	numberize = arg_numberize.getValue();
	ngramize = arg_ngramize.getValue();
	add_start_stop = arg_add_start_stop.getValue();

    // check command line arguments

    // Notes:
    // - either --words_file or --vocab_size is required.
    // - if --words_file is set,
    // - if --vocab_size is not set, it is inferred from the length of the file
    // - if --vocab_size is set, it is an error if the vocab file has a different number of lines
    // - if --numberize 0 is set and --words_file f is not set, then the output model file will not have a vocabulary, and a warning should be printed.

    // Notes:
    // - if --ngramize 0 is set, then
    // - if --ngram_size is not set, it is inferred from the training file (different from current)
    // - if --ngram_size is set, it is an error if the training file has a different n-gram size
    // - if neither --validation_file or --validation_size is set, validation will not be performed.
    // - if --numberize 0 is set, then --validation_size cannot be used.

    cerr << "Command line: " << endl;
    cerr << boost::algorithm::join(vector<string>(argv, argv+argc), " ") << endl;
	
	const string sep(" Value: ");
	cerr << arg_train_text.getDescription() << sep << arg_train_text.getValue() << endl;
	cerr << arg_train_file.getDescription() << sep << arg_train_file.getValue() << endl;
	cerr << arg_validation_text.getDescription() << sep << arg_validation_text.getValue() << endl;
	cerr << arg_validation_file.getDescription() << sep << arg_validation_file.getValue() << endl;
	cerr << arg_validation_size.getDescription() << sep << arg_validation_size.getValue() << endl;
	cerr << arg_write_words_file.getDescription() << sep << arg_write_words_file.getValue() << endl;
	cerr << arg_ngram_size.getDescription() << sep << arg_ngram_size.getValue() << endl;
	cerr << arg_vocab_size.getDescription() << sep << arg_vocab_size.getValue() << endl;
	cerr << arg_words_file.getDescription() << sep << arg_words_file.getValue() << endl;
	cerr << arg_numberize.getDescription() << sep << arg_numberize.getValue() << endl;
	cerr << arg_ngramize.getDescription() << sep << arg_ngramize.getValue() << endl;
	cerr << arg_add_start_stop.getDescription() << sep << arg_add_start_stop.getValue() << endl;
    }
    catch (TCLAP::ArgException &e)
    {
      cerr << "error: " << e.error() <<  " for arg " << e.argId() << endl;
      exit(1);
    }

    // VLF: why is this true?
    // DC: it's because the vocabulary has to be constructed from the training data only.
    // If the vocabulary is preset, we can't create the validation data.
    // - if --numberize 0 is set, then --validation_size cannot be used.
    // if (!numberize && (validation_size > 0)) {
    //     cerr <<  "Warning: without setting --numberize to 1, --validation_size cannot be used." << endl;
    // }

    // Read in training data and validation data
    vector<vector<string> > train_data;
    readSentFile(train_text, train_data);
    for (int i=0; i<train_data.size(); i++) {
        // if data is already ngramized, set/check ngram_size
        if (!ngramize) {
            if (ngram_size > 0) {
                if (ngram_size != train_data[i].size()) {
                    cerr << "Error: size of training ngrams does not match specified value of --ngram_size!" << endl;
                }
            }
            // else if --ngram_size has not been specified, set it now
            else {
                ngram_size=train_data[i].size();
            }
        }
    }
    
    vector<vector<string> > validation_data;
    if (validation_text != "") {
        readSentFile(validation_text, validation_data);
        for (int i=0; i<validation_data.size(); i++) {
	    // if data is already ngramized, set/check ngram_size
            if (!ngramize) {
                // if --ngram_size has been specified, check that it does not conflict with --ngram_size
                if (ngram_size > 0) {
                    if (ngram_size != validation_data[i].size()) {
                        cerr << "Error: size of validation ngrams does not match specified value of --ngram_size!" << endl;
                    }
                }
                // else if --ngram_size has not been specified, set it now
                else {
                    ngram_size=validation_data[i].size();
                }
            }
        }
    }
    else if (validation_size > 0)
    {
        // Create validation data
        if (validation_size > train_data.size())
	{
	    cerr << "error: requested validation size is greater than training data size" << endl;
	    exit(1);
	}
	validation_data.insert(validation_data.end(), train_data.end()-validation_size, train_data.end());
	train_data.resize(train_data.size() - validation_size);
    }

    // Construct vocabulary
    vocabulary vocab;
    int start, stop;

    // read vocabulary from file
    if (words_file != "") {
        vector<string> words;
        readWordsFile(words_file,words);
        for(vector<string>::iterator it = words.begin(); it != words.end(); ++it) {
            vocab.insert_word(*it);
        }

        // was vocab_size set? if so, verify that it does not conflict with size of vocabulary read from file
        if (vocab_size > 0) {
            if (vocab.size() != vocab_size) {
                cerr << "Error: size of vocabulary file " << vocab.size() << " != --vocab_size " << vocab_size << endl;
            }
        }
        // else, set it to the size of vocabulary read from file
        else {
            vocab_size = vocab.size();
        }

    }

    // construct vocabulary to contain top <vocab_size> most frequent words; all other words replaced by <unk>
    else {
        vocab.insert_word("<s>");
	vocab.insert_word("</s>");
	vocab.insert_word("<null>");

        // warn user that if --numberize is not set, there will be no vocabulary!
        if (!numberize) {
            cerr << "Warning: with --numberize 0 and --words_file == "", there will be no vocabulary!" << endl;
        }
        unordered_map<string,int> count;
        for (int i=0; i<train_data.size(); i++) {
            for (int j=0; j<train_data[i].size(); j++) {
                count[train_data[i][j]] += 1; 
            }
        }

        vocab.insert_most_frequent(count, vocab_size);
        if (vocab.size() < vocab_size) {
            cerr << "warning: fewer than " << vocab_size << " types in training data; the unknown word will not be learned" << endl;
        }
    }

    // write vocabulary to file
    if (write_words_file != "") {
        cerr << "Writing vocabulary to " << write_words_file << endl;
        writeWordsFile(vocab.words(), write_words_file);
    }

    // Write out numberized n-grams
    if (train_file != "")
    {
        cerr << "Writing training data to " << train_file << endl;
        writeNgrams(train_data, ngram_size, vocab, numberize, add_start_stop, ngramize, train_file);
    }
    if (validation_file != "")
    {
        cerr << "Writing validation data to " << validation_file << endl;
        writeNgrams(validation_data, ngram_size, vocab, numberize, add_start_stop, ngramize, validation_file);
    }
}
