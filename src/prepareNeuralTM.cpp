#include <iostream>
#include <vector>
#include <queue>
#include <boost/unordered_map.hpp>
#include <tclap/CmdLine.h>
#include <boost/algorithm/string/join.hpp>

using namespace std;
using namespace TCLAP;

#include "neuralLM.h" // for vocabulary
#include "util.h"

using namespace boost;
using namespace nplm;

void writeNgrams(const vector<vector<string> > &input_data, const vector<vector<string> > &output_data, int ngram_size, const vocabulary &input_vocab, const vocabulary &output_vocab, bool numberize, bool ngramize, const string &filename)
{
    ofstream file(filename.c_str());
    if (!file)
    {
	cerr << "error: could not open " << filename << endl;
	exit(1);
    }

    // check that input and output data have the same number of sentences
    if (input_data.size() != output_data.size()) {
        cerr << "Error: input and output data files have different number of lines" << endl;
        exit(1);
    }

    // for each input and output line
    int lines=input_data.size();
    if (numberize) {
        for (int i=0; i<lines; i++) {
            // convert each line to a set of ngrams
            vector<vector<int> > input_ngrams;
            vector<int> input_nums;
            for (int j=0; j<input_data[i].size(); j++) {
                input_nums.push_back(input_vocab.lookup_word(input_data[i][j]));
            }
            makeNgrams(input_nums, input_ngrams, ngram_size-1);
            
            vector<vector<int> > output_ngrams;
            vector<int> output_nums;
            for (int j=0; j<output_data[i].size(); j++) {
                output_nums.push_back(output_vocab.lookup_word(output_data[i][j]));
            }
            makeNgrams(output_nums, output_ngrams, 1);
    
            // print out cross product of input and output ngrams
            for (int j=0; j < input_ngrams.size(); j++) {
                for (int k=0; k < output_ngrams.size(); k++) {
                    int j_prime;
                    for (j_prime=0; j_prime < input_ngrams[j].size()-1; j_prime++) {
                        file << input_ngrams[j][j_prime] << " ";
                    }
                    file << input_ngrams[j][j_prime];
                    int k_prime;
                    for (k_prime=0; k_prime < output_ngrams[k].size(); k_prime++) {
                        file << " " << output_ngrams[k][k_prime];
                    }
                    file << endl;
                }
            }
        }
    }

    else {
        for (int i=0; i<lines; i++) {
            // convert each line to a set of ngrams
            vector<vector<string> > input_ngrams;
            vector<string> input_words;
            for (int j=0; j<input_data[i].size(); j++) {
                int unk = input_vocab.lookup_word("<unk>");
                // if word is unknown
                if (input_vocab.lookup_word(input_data[i][j]) == unk) {
                    input_words.push_back("<unk>");
                }
                // if word is known
                else {
                    input_words.push_back(input_data[i][j]);
                }
            }
            makeNgrams(input_words, input_ngrams, ngram_size-1);
            
            vector<vector<string> > output_ngrams;
            vector<string> output_words;
            for (int j=0; j<output_data[i].size(); j++) {
                int unk = output_vocab.lookup_word("<unk>");
                // if word is unknown
                if (output_vocab.lookup_word(output_data[i][j]) == unk) {
                    output_words.push_back("<unk>");
                }
                // if word is known
                else {
                    output_words.push_back(output_data[i][j]);
                }
            }
            makeNgrams(output_words, output_ngrams, 1);
    
            // print out cross product of input and output ngrams
            for (int j=0; j < input_ngrams.size(); j++) {
                for (int k=0; k < output_ngrams.size(); k++) {
                    int j_prime;
                    for (j_prime=0; j_prime < input_ngrams[j].size()-1; j_prime++) {
                        file << input_ngrams[j][j_prime] << " ";
                    }
                    file << input_ngrams[j][j_prime];
                    int k_prime;
                    for (k_prime=0; k_prime < output_ngrams[k].size(); k_prime++) {
                        file << " " << output_ngrams[k][k_prime];
                    }
                    file << endl;
                }
            }
        }
    }
    file.close();
}
    
int main(int argc, char *argv[])
{
    int ngram_size, input_vocab_size, output_vocab_size, validation_size;
    bool add_start_stop, numberize, ngramize;
    string input_train_text, output_train_text, train_file, input_validation_text, output_validation_text, validation_file, write_input_words_file, write_output_words_file, input_words_file, output_words_file;

    try
    {
	CmdLine cmd("Prepares training data for training a language model.", ' ', "0.1");

	// The options are printed in reverse order
    
    ValueArg<bool> arg_ngramize("", "ngramize", "If true, convert lines to ngrams. Default: true.", false, true, "bool", cmd);
    ValueArg<bool> arg_numberize("", "numberize", "If true, convert words to numbers. Default: true.", false, true, "bool", cmd);
    ValueArg<bool> arg_add_start_stop("", "add_start_stop", "If true, prepend (ngram_size-1) start symbols and postpend 1 stop symbol. Default: true.", false, true, "bool", cmd);
    ValueArg<int> arg_input_vocab_size("", "input_vocab_size", "Vocabulary size.", false, -1, "int", cmd);
    ValueArg<int> arg_output_vocab_size("", "output_vocab_size", "Vocabulary size.", false, -1, "int", cmd);
    ValueArg<string> arg_input_words_file("", "input_words_file", "File specifying words that should be included in vocabulary; all other words will be replaced by <unk>.", false, "", "string", cmd);
    ValueArg<string> arg_output_words_file("", "output_words_file", "File specifying words that should be included in vocabulary; all other words will be replaced by <unk>.", false, "", "string", cmd);
    ValueArg<int> arg_ngram_size("", "ngram_size", "Size of n-grams.", true, -1, "int", cmd);
	ValueArg<string> arg_write_input_words_file("", "write_input_words_file", "Output vocabulary.", false, "", "string", cmd);
	ValueArg<string> arg_write_output_words_file("", "write_output_words_file", "Output vocabulary.", false, "", "string", cmd);
    ValueArg<int> arg_validation_size("", "validation_size", "How many lines from training data to hold out for validation. Default: 0.", false, 0, "int", cmd);
	ValueArg<string> arg_validation_file("", "validation_file", "Output validation data (numberized n-grams).", false, "", "string", cmd);
	ValueArg<string> arg_input_validation_text("", "input_validation_text", "Input validation data (tokenized). Overrides --validation_size. Default: none.", false, "", "string", cmd);
	ValueArg<string> arg_output_validation_text("", "output_validation_text", "Input validation data (tokenized). Overrides --validation_size. Default: none.", false, "", "string", cmd);
	ValueArg<string> arg_train_file("", "train_file", "Output training data (numberized n-grams).", false, "", "string", cmd);
    ValueArg<string> arg_input_train_text("", "input_train_text", "Input training data (tokenized).", true, "", "string", cmd);
    ValueArg<string> arg_output_train_text("", "output_train_text", "Input training data (tokenized).", true, "", "string", cmd);

	cmd.parse(argc, argv);

	input_train_text = arg_input_train_text.getValue();
	output_train_text = arg_output_train_text.getValue();
	train_file = arg_train_file.getValue();
	validation_file = arg_validation_file.getValue();
	input_validation_text = arg_input_validation_text.getValue();
	output_validation_text = arg_output_validation_text.getValue();
	input_validation_text = arg_input_validation_text.getValue();
	output_validation_text = arg_output_validation_text.getValue();
	validation_size = arg_validation_size.getValue();
	write_input_words_file = arg_write_input_words_file.getValue();
	write_output_words_file = arg_write_output_words_file.getValue();
	ngram_size = arg_ngram_size.getValue();
	input_vocab_size = arg_input_vocab_size.getValue();
	output_vocab_size = arg_output_vocab_size.getValue();
	input_words_file = arg_input_words_file.getValue();
	output_words_file = arg_output_words_file.getValue();
	numberize = arg_numberize.getValue();
	ngramize = arg_ngramize.getValue();
	add_start_stop = arg_add_start_stop.getValue();

    // check command line arguments

    // Notes:
    // - either --words_file or --vocab_size is required.
    // - if --words_file is set,
    // - if --vocab_size is not set, it is inferred from the length of the file
    // - if --vocab_size is set, it is an error if the vocab file has a different number of lines
    // - if --numberize 0 is set and --use_vocab f is not set, then the output model file will not have a vocabulary, and a warning should be printed.
    if ((input_words_file == "") && (input_vocab_size == -1)) {
        cerr << "Error: either --input_words_file or --input_vocab_size is required." << endl;
        exit(1);
    }
    if ((output_words_file == "") && (output_vocab_size == -1)) {
        cerr << "Error: either --output_words_file or --output_vocab_size is required." << endl;
        exit(1);
    }

    // Notes:
    // - if --ngramize 0 is set, then
    // - if --ngram_size is not set, it is inferred from the training file (different from current)
    // - if --ngram_size is set, it is an error if the training file has a different n-gram size
    // - if neither --validation_file or --validation_size is set, validation will not be performed.
    // - if --numberize 0 is set, then --validation_size cannot be used.

    cerr << "Command line: " << endl;
    cerr << boost::algorithm::join(vector<string>(argv, argv+argc), " ") << endl;
	
	const string sep(" Value: ");
	cerr << arg_input_train_text.getDescription() << sep << arg_input_train_text.getValue() << endl;
	cerr << arg_output_train_text.getDescription() << sep << arg_output_train_text.getValue() << endl;
	cerr << arg_train_file.getDescription() << sep << arg_train_file.getValue() << endl;
	cerr << arg_input_validation_text.getDescription() << sep << arg_input_validation_text.getValue() << endl;
	cerr << arg_output_validation_text.getDescription() << sep << arg_output_validation_text.getValue() << endl;
	cerr << arg_validation_file.getDescription() << sep << arg_validation_file.getValue() << endl;
	cerr << arg_validation_size.getDescription() << sep << arg_validation_size.getValue() << endl;
	cerr << arg_write_input_words_file.getDescription() << sep << arg_write_input_words_file.getValue() << endl;
	cerr << arg_write_output_words_file.getDescription() << sep << arg_write_output_words_file.getValue() << endl;
	cerr << arg_ngram_size.getDescription() << sep << arg_ngram_size.getValue() << endl;
	cerr << arg_input_vocab_size.getDescription() << sep << arg_input_vocab_size.getValue() << endl;
	cerr << arg_output_vocab_size.getDescription() << sep << arg_output_vocab_size.getValue() << endl;
	cerr << arg_input_words_file.getDescription() << sep << arg_input_words_file.getValue() << endl;
	cerr << arg_output_words_file.getDescription() << sep << arg_output_words_file.getValue() << endl;
	cerr << arg_numberize.getDescription() << sep << arg_numberize.getValue() << endl;
	cerr << arg_ngramize.getDescription() << sep << arg_ngramize.getValue() << endl;
	cerr << arg_add_start_stop.getDescription() << sep << arg_add_start_stop.getValue() << endl;
    }
    catch (TCLAP::ArgException &e)
    {
      cerr << "error: " << e.error() <<  " for arg " << e.argId() << endl;
      exit(1);
    }

    // Read in input training data and validation data
    vector<vector<string> > input_train_data;
    readSentFile(input_train_text, input_train_data);
    if (add_start_stop) {
      for (int i=0; i<input_train_data.size(); i++) {
	vector<string> input_train_data_start_stop;
	addStartStop<string>(input_train_data[i], input_train_data_start_stop, ngram_size, "<s>", "</s>");
	input_train_data[i]=input_train_data_start_stop;
      }
    }
    
    vector<vector<string> > input_validation_data;
    if (input_validation_text != "") {
        readSentFile(input_validation_text, input_validation_data);
        if (add_start_stop) {
	  for (int i=0; i<input_validation_data.size(); i++) {
	    vector<string> input_validation_data_start_stop;
	    addStartStop<string>(input_validation_data[i], input_validation_data_start_stop, ngram_size, "<s>", "</s>");
	    input_validation_data[i]=input_validation_data_start_stop;
	  }
        }
    }
    else if (validation_size > 0)
    {
        if (validation_size > input_train_data.size())
	{
	    cerr << "error: requested input_validation size is greater than training data size" << endl;
	    exit(1);
	}
	input_validation_data.insert(input_validation_data.end(), input_train_data.end()-validation_size, input_train_data.end());
	input_train_data.resize(input_train_data.size() - validation_size);
    }

    // Read in output training data and validation data
    vector<vector<string> > output_train_data;
    readSentFile(output_train_text, output_train_data);
    if (add_start_stop) {
      for (int i=0; i<output_train_data.size(); i++) {
	vector<string> output_train_data_start_stop;
	addStartStop<string>(output_train_data[i], output_train_data_start_stop, 1, "<s>", "</s>");
	output_train_data[i]=output_train_data_start_stop;
      }
    }
    
    vector<vector<string> > output_validation_data;
    if (output_validation_text != "") {
        readSentFile(output_validation_text, output_validation_data);
        if (add_start_stop) {
	  for (int i=0; i<output_validation_data.size(); i++) {
	    vector<string> output_validation_data_start_stop;
	    addStartStop<string>(output_validation_data[i], output_validation_data_start_stop, 1, "<s>", "</s>");
	    output_validation_data[i]=output_validation_data_start_stop;
	  }
        }
    }
    else if (validation_size > 0)
    {
        if (validation_size > output_train_data.size())
	{
	    cerr << "error: requested output_validation size is greater than training data size" << endl;
	    exit(1);
	}
	output_validation_data.insert(output_validation_data.end(), output_train_data.end()-validation_size, output_train_data.end());
	output_train_data.resize(output_train_data.size() - validation_size);
    }

    // Construct input vocabulary
    vocabulary input_vocab;
    int input_start = input_vocab.insert_word("<s>");
    int input_stop = input_vocab.insert_word("</s>");
    input_vocab.insert_word("<null>");

    // read input vocabulary from file
    if (input_words_file != "") {
        vector<string> words;
        readWordsFile(input_words_file,words);
        for(vector<string>::iterator it = words.begin(); it != words.end(); ++it) {
            input_vocab.insert_word(*it);
        }
        // was input_vocab_size set? if so, verify that it does not conflict with size of vocabulary read from file
        if (input_vocab_size > 0) {
            if (input_vocab.size() != input_vocab_size) {
                cerr << "Error: size of input_vocabulary file " << input_vocab.size() << " != --input_vocab_size " << input_vocab_size << endl;
            }
        }
        // else, set it to the size of vocabulary read from file
        else {
            input_vocab_size = input_vocab.size();
        }
    }

    // or construct input vocabulary to contain top <input_vocab_size> most frequent words; all other words replaced by <unk>
    else {
        unordered_map<string,int> count;
        for (int i=0; i<input_train_data.size(); i++) {
            for (int j=0; j<input_train_data[i].size(); j++) {
                count[input_train_data[i][j]] += 1; 
            }
        }

        input_vocab.insert_most_frequent(count, input_vocab_size);
        if (input_vocab.size() < input_vocab_size) {
            cerr << "warning: fewer than " << input_vocab_size << " types in training data; the unknown word will not be learned" << endl;
        }
    }

    // Construct output vocabulary
    vocabulary output_vocab;
    int output_start = output_vocab.insert_word("<s>");
    int output_stop = output_vocab.insert_word("</s>");
    output_vocab.insert_word("<null>");

    // read output vocabulary from file
    if (output_words_file != "") {
        vector<string> words;
        readWordsFile(output_words_file,words);
        for(vector<string>::iterator it = words.begin(); it != words.end(); ++it) {
            output_vocab.insert_word(*it);
        }
        // was output_vocab_size set? if so, verify that it does not conflict with size of vocabulary read from file
        if (output_vocab_size > 0) {
            if (output_vocab.size() != output_vocab_size) {
                cerr << "Error: size of output_vocabulary file " << output_vocab.size() << " != --output_vocab_size " << output_vocab_size << endl;
            }
        }
        // else, set it to the size of vocabulary read from file
        else {
            output_vocab_size = output_vocab.size();
        }
    }

    // or construct output vocabulary to contain top <output_vocab_size> most frequent words; all other words replaced by <unk>
    else {
        unordered_map<string,int> count;
        for (int i=0; i<output_train_data.size(); i++) {
            for (int j=0; j<output_train_data[i].size(); j++) {
                count[output_train_data[i][j]] += 1; 
            }
        }

        output_vocab.insert_most_frequent(count, output_vocab_size);
        if (output_vocab.size() < output_vocab_size) {
            cerr << "warning: fewer than " << output_vocab_size << " types in training data; the unknown word will not be learned" << endl;
        }
    }

    // write input vocabulary to file
    if (write_input_words_file != "") {
        cerr << "Writing vocabulary to " << write_input_words_file << endl;
        writeWordsFile(input_vocab.words(), write_input_words_file);
    }

    // write output vocabulary to file
    if (write_output_words_file != "") {
        cerr << "Writing vocabulary to " << write_output_words_file << endl;
        writeWordsFile(output_vocab.words(), write_output_words_file);
    }

    // Write out input and output numberized n-grams
    if (train_file != "")
    {
        cerr << "Writing training data to " << train_file << endl;
        writeNgrams(input_train_data, output_train_data, ngram_size, input_vocab, output_vocab, numberize, ngramize, train_file);

    }
    if (validation_file != "")
    {
        cerr << "Writing validation data to " << validation_file << endl;
        writeNgrams(input_validation_data, output_validation_data, ngram_size, input_vocab, output_vocab, numberize, ngramize, validation_file);
    }
}
