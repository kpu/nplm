#ifndef NETWORK_H
#define NETWORK_H

#include "neuralClasses.h"
#include "util.h"

namespace nplm
{

// is this cheating?
using Eigen::Matrix;
using Eigen::MatrixBase;
using Eigen::Dynamic;

class propagator {
    int minibatch_size;
    const model *pnn;

public:
    Node<Input_word_embeddings> input_layer_node;
    Node<Linear_layer> first_hidden_linear_node;
    Node<Activation_function> first_hidden_activation_node;
    Node<Linear_layer> second_hidden_linear_node;
    Node<Activation_function> second_hidden_activation_node;
    Node<Output_word_embeddings> output_layer_node;

public:
    propagator () : minibatch_size(0), pnn(0) { }

    propagator (const model &nn, int minibatch_size)
      :
        pnn(&nn),
        input_layer_node(&nn.input_layer, minibatch_size),
	first_hidden_linear_node(&nn.first_hidden_linear, minibatch_size),
	first_hidden_activation_node(&nn.first_hidden_activation, minibatch_size),
        second_hidden_linear_node(&nn.second_hidden_linear, minibatch_size),
	second_hidden_activation_node(&nn.second_hidden_activation, minibatch_size),
	output_layer_node(&nn.output_layer, minibatch_size),
	minibatch_size(minibatch_size)
    {
    }

    // This must be called if the underlying model is resized.
    void resize(int minibatch_size) {
      this->minibatch_size = minibatch_size;
      input_layer_node.resize(minibatch_size);
      first_hidden_linear_node.resize(minibatch_size);
      first_hidden_activation_node.resize(minibatch_size);
      second_hidden_linear_node.resize(minibatch_size);
      second_hidden_activation_node.resize(minibatch_size);
      output_layer_node.resize(minibatch_size);
    }

    void resize() { resize(minibatch_size); }

    template <typename Derived>
    void fProp(const MatrixBase<Derived> &data)
    {
        if (!pnn->premultiplied)
	{
            start_timer(0);
	    input_layer_node.param->fProp(data, input_layer_node.fProp_matrix);
	    stop_timer(0);
	    
	    start_timer(1);
	    first_hidden_linear_node.param->fProp(input_layer_node.fProp_matrix, 
						  first_hidden_linear_node.fProp_matrix);
	} 
	else
	{
	    int n_inputs = first_hidden_linear_node.param->n_inputs();
	    USCMatrix<double> sparse_data;
	    input_layer_node.param->munge(data, sparse_data);

	    start_timer(1);
	    first_hidden_linear_node.param->fProp(sparse_data,
						  first_hidden_linear_node.fProp_matrix);
	}
	first_hidden_activation_node.param->fProp(first_hidden_linear_node.fProp_matrix,
						  first_hidden_activation_node.fProp_matrix);
	stop_timer(1);
    

	start_timer(2);
	second_hidden_linear_node.param->fProp(first_hidden_activation_node.fProp_matrix,
					       second_hidden_linear_node.fProp_matrix);
	second_hidden_activation_node.param->fProp(second_hidden_linear_node.fProp_matrix,
						   second_hidden_activation_node.fProp_matrix);
	stop_timer(2);

	// The propagation stops here because the last layer is very expensive.
    }

    // Dense version (for standard log-likelihood)
    template <typename DerivedIn, typename DerivedOut>
    void bProp(const MatrixBase<DerivedIn> &data,
	       const MatrixBase<DerivedOut> &output,
	       double learning_rate, double momentum, double L2_reg) 
    {
        // Output embedding layer

        start_timer(7);
        output_layer_node.param->bProp(output,
				       output_layer_node.bProp_matrix);
	stop_timer(7);
	
	start_timer(8);
	output_layer_node.param->computeGradient(second_hidden_activation_node.fProp_matrix,
						 output,
						 learning_rate, momentum);
	stop_timer(8);

	bPropRest(data, learning_rate, momentum, L2_reg);
    }

    // Sparse version (for NCE log-likelihood)
    template <typename DerivedIn, typename DerivedOutI, typename DerivedOutV>
    void bProp(const MatrixBase<DerivedIn> &data,
	       const MatrixBase<DerivedOutI> &samples, const MatrixBase<DerivedOutV> &weights,
	       double learning_rate, double momentum, double L2_reg) 
    {

        // Output embedding layer

        start_timer(7);
        output_layer_node.param->bProp(samples, weights, 
				       output_layer_node.bProp_matrix);
	stop_timer(7);
	

	start_timer(8);
	output_layer_node.param->computeGradient(second_hidden_activation_node.fProp_matrix,
						 samples, weights,
						 learning_rate, momentum);
	stop_timer(8);

	bPropRest(data, learning_rate, momentum, L2_reg);
    }

private:
    template <typename DerivedIn>
    void bPropRest(const MatrixBase<DerivedIn> &data,
		   double learning_rate, double momentum, double L2_reg) 
    {
	// Second hidden layer

        start_timer(9);
	second_hidden_activation_node.param->bProp(output_layer_node.bProp_matrix,
						   second_hidden_activation_node.bProp_matrix,
						   second_hidden_linear_node.fProp_matrix,
						   second_hidden_activation_node.fProp_matrix);

	second_hidden_linear_node.param->bProp(second_hidden_activation_node.bProp_matrix,
					       second_hidden_linear_node.bProp_matrix);
	stop_timer(9);

	start_timer(10);
	second_hidden_linear_node.param->computeGradient(second_hidden_activation_node.bProp_matrix,
							 first_hidden_activation_node.fProp_matrix,
							 learning_rate, momentum, L2_reg);
	stop_timer(10);

	// First hidden layer

	start_timer(11);
	first_hidden_activation_node.param->bProp(second_hidden_linear_node.bProp_matrix,
						  first_hidden_activation_node.bProp_matrix,
						  first_hidden_linear_node.fProp_matrix,
						  first_hidden_activation_node.fProp_matrix);

	first_hidden_linear_node.param->bProp(first_hidden_activation_node.bProp_matrix,
					      first_hidden_linear_node.bProp_matrix);
	stop_timer(11);
	
	start_timer(12);
	first_hidden_linear_node.param->computeGradient(first_hidden_activation_node.bProp_matrix,
							input_layer_node.fProp_matrix,
							learning_rate, momentum, L2_reg);
	stop_timer(12);

	// Input word embeddings
	
	start_timer(13);
	input_layer_node.param->computeGradient(first_hidden_linear_node.bProp_matrix,
						data,
						learning_rate, momentum, L2_reg);
	stop_timer(13);

    }
};

} // namespace nplm

#endif
