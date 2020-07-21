#include<Rcpp.h>
#include <vector>
#include <math.h>
#include "../inst/include/dense.h"

using namespace std;

/***********************************/
Dense::Dense(unsigned long _n_neurons, string _activation /*=sigmoid*/, double learning_rate_init_ /*= 0.01*/, bool bias_ /*= 1*/,  const string & alg /*= "sgd"*/)
    :n_neurons (_n_neurons),
    activation (_activation),
    learning_rate_init (learning_rate_init_),
    bias (bias_),
    output_layer (false),
    algo (alg),
    beta1 (0.9),
    beta2 (0.99)
{
    /* Allocate memory for the vectors of the layer */
    this->net. resize (_n_neurons, 0);
    this->E. resize (_n_neurons, 0);
    this->O. resize (_n_neurons, 0);

    this->alpha. resize (_n_neurons);
    this->M. resize (_n_neurons);
    this->V. resize (_n_neurons);

    this->W. resize (_n_neurons);
    this->DeltaW. resize (_n_neurons);

}

/***********************************/
void Dense::set_input_dim (vector<unsigned long> in_dim)
{
    if (in_dim. size () > 1)
    {
        Rcpp::Rcout << "Error in input dimension for Dense layer, the input should be a 1D vector.\n";
        Rcpp::stop ("\n.");
    }
    this->input_dim = in_dim[0];
    // set uniform weights for each neuron

    // init alpha parameters (including adam parameters)
    for (unsigned long i = 0; i < n_neurons; ++i)
    {
        alpha[i]. resize (input_dim + unsigned (this->bias), learning_rate_init);
        M[i]. resize (input_dim + unsigned (this->bias), 0);
        V[i]. resize (input_dim + unsigned (this->bias), 0);
        DeltaW[i]. resize (input_dim + unsigned (this->bias), 0);
        W[i] =  random_vector (input_dim + unsigned (this->bias));
    }
}

bool Dense::contains_bias() {return bias;}

bool Dense::is_output(){return output_layer;}

void Dense::set_output_layer(bool last){output_layer = last;}

vector<unsigned long> Dense::get_output_dim(){return {n_neurons};}

vector<unsigned long> Dense::get_input_dim(){return {input_dim};}
/***********************************/
MatD Dense::simulate (const MatD & input_, bool store)
{
    if (input_. size () > 1)
    {
        Rcpp::Rcout << "Input of the dense layer is not correct. Matrix of 1 row is required. \n";
        Rcpp::Rcout << "The input matrix contains " << input_. size () << ".\n";
        Rcpp::stop ("\n.");
    }

    vector<double> output;
    vector<double> input__ ( input_[0]);

    if (this->bias)
        input__.insert (input__. begin (), 1);

    // output without activation function
    //MultCVDouble (this->W, input, output);
    output = matrix_dot (this->W, input__);

    if (store)
    {
        net = VectD (output);
        input = VectD (input__);
    }
    // output with activation function
    output = vect_activation (output, activation);

    if (store)
        this->O = VectD (output);

    return {output};
}

/***********************************/
void Dense::computeErrors(const MatD & nextErrors)
{
    if (nextErrors. size () > 1)
    {
        Rcpp::Rcout << "Error to backpropagate to the dense layer is not correct. Matrix of 1 row is required. \n";
        Rcpp::Rcout << "The output errors matrix contains " << nextErrors. size () << ".\n";
        Rcpp::stop ("\n.");
    }

    /* nextErrors: errors of the next layer, and if the layer is an output layer, it is : (expected out - output) */
    if (nextErrors[0]. size () != this->n_neurons)
    {
        Rcpp::Rcout << "Error in computing the error, output dimensions are not correct.\n";
        Rcpp::Rcout << "Expecting " <<  this->n_neurons << " as output dimensions \n";
        Rcpp::Rcout << "While, the given errors are of size: " << nextErrors. size ();
    }

    this->E. clear ();
    this->E. resize (this->n_neurons);

    VectD diff_net =  diff_activation (net, activation);

    E = matrix_dot (nextErrors[0], diff_net);
}

/***********************************/
void Dense::updateWeights (unsigned long numb_iter)
{
    //vector<double> prevOutput (previousOutput);

    //copy_vector (prevOutput, previousOutput);

    //if (this->bias)
        //prevOutput. insert (prevOutput. begin (), 1);

    double momentum;
    if (algo == "sgd")
         momentum = 0.9;
    else
         momentum = 0.0;

    for (unsigned j = 0; j < this->n_neurons; ++j)
        for (unsigned i = 0; i < this->W[0]. size (); ++i)
        {
            this->DeltaW[j][i] =  momentum * this->DeltaW[j][i] + (1 - momentum) * input[i] * this->E[j];
            this->W[j][i] -= alpha[j][i]  * this->DeltaW[j][i];
        }

    // update learning rate
    if (algo == "adam")
    {
        for (unsigned j = 0; j < this->n_neurons; ++j)
             for (unsigned i = 0; i < alpha[0]. size (); ++i)
                {
                    M[j][i] = ((beta1 * M[j][i]) + ((1 - beta1) * DeltaW[j][i])) ;
                    V[j][i] = ((beta2 * V[j][i]) + ((1 - beta2) * DeltaW[j][i] * DeltaW[j][i])) ;

                    M[j][i] /= (1 - pow (beta1, numb_iter + 1));
                    V[j][i] /= (1 - pow (beta2, numb_iter + 1));

                    alpha[j][i] = alpha[j][i] - (M[j][i] * 0.001 / (sqrt (V[j][i]) + 0.00000001));
                }
    }
}

/*********************************/
// return the errors to backpropagate to the previous layer
MatD Dense::get_errors ()
{
    VectD A = matrix_dot (Transpose (this->W), this->E);

    if (this->bias == 1)
        A. erase (A. begin ());
    return {A};
}

MatD Dense::get_weights(){return this->W;}

string Dense::getType(){return "dense";}
