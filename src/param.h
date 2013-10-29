#pragma once

#include <string>

namespace nplm
{

struct param 
{
    std::string train_file;
    std::string validation_file;
    std::string test_file;

    std::string model_file;

    std::string unigram_probs_file;
    std::string words_file;
    std::string input_words_file;
    std::string output_words_file;
    std::string model_prefix;

    int ngram_size;
    int vocab_size;
    int input_vocab_size;
    int output_vocab_size;
    int num_hidden;
    int embedding_dimension;
    int input_embedding_dimension;
    int output_embedding_dimension;
    std::string activation_function;
    std::string loss_function;

    int minibatch_size;
    int validation_minibatch_size;
    int num_epochs;
    double learning_rate;

    bool init_normal;
    double init_range;

    int num_noise_samples;

    bool use_momentum;
    double initial_momentum;
    double final_momentum;

    double L2_reg;

    bool normalization;
    double normalization_init;

    int num_threads;
  
    bool share_embeddings;

};

} // namespace nplm
