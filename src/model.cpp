#include <cstdlib>
#include <iostream>
#include <boost/lexical_cast.hpp>

#include "model.h"
#include "param.h"

using namespace std;
using namespace boost;
using namespace boost::random;

namespace nplm
{

    void model::resize(int ngram_size,
        int input_vocab_size,
        int output_vocab_size,
        int input_embedding_dimension,
        int num_hidden,
        int output_embedding_dimension)
{
    input_layer.resize(input_vocab_size, input_embedding_dimension, ngram_size-1);
    first_hidden_linear.resize(num_hidden, input_embedding_dimension*(ngram_size-1));
    first_hidden_activation.resize(num_hidden);
    second_hidden_linear.resize(output_embedding_dimension, num_hidden);
    second_hidden_activation.resize(output_embedding_dimension);
    output_layer.resize(output_vocab_size, output_embedding_dimension);
    this->ngram_size = ngram_size;
    this->input_vocab_size = input_vocab_size;
    this->output_vocab_size = output_vocab_size;
    this->input_embedding_dimension = input_embedding_dimension;
    this->num_hidden = num_hidden;
    this->output_embedding_dimension = output_embedding_dimension;
    premultiplied = false;
}
  
void model::initialize(mt19937 &init_engine, bool init_normal, double init_range, double init_bias)
{
    input_layer.initialize(init_engine, init_normal, init_range);
    output_layer.initialize(init_engine, init_normal, init_range, init_bias);
    first_hidden_linear.initialize(init_engine, init_normal, init_range);
    second_hidden_linear.initialize(init_engine, init_normal, init_range);
}

void model::premultiply()
{
    // Since input and first_hidden_linear are both linear,
    // we can multiply them into a single linear layer *if* we are not training
    int context_size = ngram_size-1;
    Matrix<double,Dynamic,Dynamic> U = first_hidden_linear.U;
    first_hidden_linear.U.resize(num_hidden, input_vocab_size * context_size);
    for (int i=0; i<context_size; i++)
        first_hidden_linear.U.middleCols(i*input_vocab_size, input_vocab_size) = U.middleCols(i*input_embedding_dimension, input_embedding_dimension) * input_layer.W->transpose();
    input_layer.W->resize(1,1); // try to save some memory
    premultiplied = true;
}

void model::readConfig(ifstream &config_file)
{
    string line;
    vector<string> fields;
    int ngram_size, vocab_size, input_embedding_dimension, num_hidden, output_embedding_dimension;
    activation_function_type activation_function = this->activation_function;
    while (getline(config_file, line) && line != "")
    {
        splitBySpace(line, fields);
	if (fields[0] == "ngram_size")
	    ngram_size = lexical_cast<int>(fields[1]);
	else if (fields[0] == "vocab_size")
	    input_vocab_size = output_vocab_size = lexical_cast<int>(fields[1]);
	else if (fields[0] == "input_vocab_size")
	    input_vocab_size = lexical_cast<int>(fields[1]);
	else if (fields[0] == "output_vocab_size")
	    output_vocab_size = lexical_cast<int>(fields[1]);
	else if (fields[0] == "input_embedding_dimension")
	    input_embedding_dimension = lexical_cast<int>(fields[1]);
	else if (fields[0] == "num_hidden")
	    num_hidden = lexical_cast<int>(fields[1]);
	else if (fields[0] == "output_embedding_dimension")
	    output_embedding_dimension = lexical_cast<int>(fields[1]);
	else if (fields[0] == "activation_function")
	    activation_function = string_to_activation_function(fields[1]);
	else if (fields[0] == "version")
	{
	    int version = lexical_cast<int>(fields[1]);
	    if (version != 1)
	    {
		cerr << "error: file format mismatch (expected 1, found " << version << ")" << endl;
		exit(1);
	    }
	}
	else
	    cerr << "warning: unrecognized field in config: " << fields[0] << endl;
    }
    resize(ngram_size,
        input_vocab_size,
        output_vocab_size,
        input_embedding_dimension,
        num_hidden,
        output_embedding_dimension);
    set_activation_function(activation_function);
}

void model::readConfig(const string &filename)
{
    ifstream config_file(filename.c_str());
    if (!config_file)
    {
        cerr << "error: could not open config file " << filename << endl;
	exit(1);
    }
    readConfig(config_file);
    config_file.close();
}
 
void model::read(const string &filename)
{
    vector<string> input_words;
    vector<string> output_words;
    read(filename, input_words, output_words);
}

void model::read(const string &filename, vector<string> &input_words, vector<string> &output_words)
{
    ifstream file(filename.c_str());
    if (!file) throw runtime_error("Could not open file " + filename);
    
    param myParam;
    string line;
    
    while (getline(file, line))
    {
	if (line == "\\config")
	{
	    readConfig(file);
	}

	else if (line == "\\vocab")
	{
	    input_words.clear();
	    readWordsFile(file, input_words);
	    output_words = input_words;
	}

	else if (line == "\\input_vocab")
	{
	    input_words.clear();
	    readWordsFile(file, input_words);
	}

	else if (line == "\\output_vocab")
	{
	    output_words.clear();
	    readWordsFile(file, output_words);
	}

	else if (line == "\\input_embeddings")
	    input_layer.read(file);
	else if (line == "\\hidden_weights 1")
	    first_hidden_linear.read(file);
	else if (line == "\\hidden_weights 2")
	    second_hidden_linear.read(file);
	else if (line == "\\output_weights")
	    output_layer.read_weights(file);
	else if (line == "\\output_biases")
	    output_layer.read_biases(file);
	else if (line == "\\end")
	    break;
	else if (line == "")
	    continue;
	else
	{
	    cerr << "warning: unrecognized section: " << line << endl;
	    // skip over section
	    while (getline(file, line) && line != "") { }
	}
    }
    file.close();
}

    void model::write(const string &filename, const vector<string> &input_words, const vector<string> &output_words)
{ 
    write(filename, &input_words, &output_words);
}

void model::write(const string &filename) 
{ 
    write(filename, NULL, NULL);
}

    void model::write(const string &filename, const vector<string> *input_pwords, const vector<string> *output_pwords)
{
    ofstream file(filename.c_str());
    if (!file) throw runtime_error("Could not open file " + filename);
    
    file << "\\config" << endl;
    file << "version 1" << endl;
    file << "ngram_size " << ngram_size << endl;
    file << "input_vocab_size " << input_vocab_size << endl;
    file << "output_vocab_size " << output_vocab_size << endl;
    file << "input_embedding_dimension " << input_embedding_dimension << endl;
    file << "num_hidden " << num_hidden << endl;
    file << "output_embedding_dimension " << output_embedding_dimension << endl;
    file << "activation_function " << activation_function_to_string(activation_function) << endl;
    file << endl;
    
    if (input_pwords)
    {
        file << "\\input_vocab" << endl;
	writeWordsFile(*input_pwords, file);
	file << endl;
    }

    if (output_pwords)
    {
        file << "\\output_vocab" << endl;
	writeWordsFile(*output_pwords, file);
	file << endl;
    }

    file << "\\input_embeddings" << endl;
    input_layer.write(file);
    file << endl;
    
    file << "\\hidden_weights 1" << endl;
    first_hidden_linear.write(file);
    file << endl;
    
    file << "\\hidden_weights 2" << endl;
    second_hidden_linear.write(file);
    file << endl;
    
    file << "\\output_weights" << endl;
    output_layer.write_weights(file);
    file << endl;
    
    file << "\\output_biases" << endl;
    output_layer.write_biases(file);
    file << endl;
    
    file << "\\end" << endl;
    file.close();
}


} // namespace nplm
