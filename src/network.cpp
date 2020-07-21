/*
  This file is part of NlinTS. NlinTS is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 2 of the License, or
  (at your option) any later version.
  NlinTS is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
*/

#include<Rcpp.h>
#include <iomanip>
#include<math.h>
#include<random>
#include "../inst/include/network.h"

/******** add a layer to the network **************/
Network::Network(vector<unsigned long> _input_dim):
    input_dim (_input_dim),
    nb_layers (0)
{}

Network::Network(){}

void Network::set_input_size(vector<unsigned long> _input_dim)
{
    input_dim = vector<unsigned long> (_input_dim);
}

Network::~Network()
{
    for (auto layer : this->layers)
        delete layer;
}

void Network::addLayer(Layer * layer)
{
    vector<unsigned long> in_dim;

    if (this->layers. size () == 0)
        in_dim = vector<unsigned long> (input_dim);
        // copy_vector (in_dim, this->input_dim);
    else
        in_dim = vector<unsigned long> ( this->layers.back ()->get_output_dim ());
        //copy_vector (in_dim, this->layers.back ()->get_output_dim ());

    // add the layer
    this->layers.push_back (layer);

    // set input dimension
    this->layers.back ()-> set_input_dim (in_dim);

    // consider this layer as output layer
    this->layers. back ()->set_output_layer(true);

    // consider the previous layer as hidden layer
    if (this->layers. size () > 1)
        this->layers [this->layers. size () - 2]-> set_output_layer (false);

     nb_layers ++;
}

/******** Propagate an input through the signal **************/
MatD Network::simulate(const MatD & _input, bool store)
{
    MatD signal (_input);

    for (Layer* layer : this->layers)
        signal = layer-> simulate (signal, store);

    return (signal);
}

/*********************************************************/
void Network::backpropagation(const VectD & output)
{
    MatD propagatedErrors;
    int num_layers = int(this->layers.size ());

    for (int i = num_layers - 1; i >= 0; --i)
    {
        // hidden layer
        if (i < (num_layers - 1))
        {
            propagatedErrors = this->layers[i+1]-> get_errors ();
        }
        // Output layer
        else
        {
            propagatedErrors = this->layers[i]-> get_output();
            for (unsigned j = 0; j < propagatedErrors. size (); ++j)
                propagatedErrors[j][0] -= output[j];
        }

        // packpropagate the errors through the layer
        this->layers[i]->computeErrors (propagatedErrors);
        propagatedErrors. clear ();
    }
}

/*******************************************************/
void Network::updateWeight (unsigned long numb_iter)
{
    for (unsigned i = 0; i < this->layers. size (); ++i)
    {
        this->layers[i]->updateWeights (numb_iter);
    }
}

/************* loss function (mse) *******************/
double Network::loss (const VectD & preds, const VectD & real)
{
    if (preds. size () != real. size ())
    {
        Rcpp::Rcout << "Error in calculating the loss function, preds and real have not the same size. \n";
        Rcpp::stop ("\n.");
    }
    double mse = 0;
    for (unsigned i = 0; i < preds. size (); ++i)
        mse += (real [i] - preds[i])*(real [i] - preds[i]);

    return mse / preds. size ();
}

/************* average loss function *******************/
double Network::average_loss(const MatD & preds, const MatD & real)
{
    if (preds. size () != real. size ())
    {
        Rcpp::Rcout << "Error in calculating the average_loss function, preds and real have not the same size. \n";
        Rcpp::stop ("\n.");
    }

    double avg_mse = 0;
     for (unsigned i = 0; i < preds. size (); ++i)
         avg_mse += loss (preds[i], real [i]);

     return avg_mse /  preds. size ();
}

/************** Train the network : one epoch *********************/
void Network::train(const MatD & X, const MatD & y)
{
    MatD predictions;
    unsigned long i = 0;
    for (const auto & row : X)
    {
        predictions. push_back (this->simulate ({row}, true)[0]);
        backpropagation (y[i]);
        updateWeight (i);
        ++i;
    }
    //Rcpp::Rcout << "average mse: " << average_loss (predictions, y) << endl;
}
/************** Train the network from a vector of matrix *********************/
/*void Network::train(const vector<MatD> &X, const MatD &y)
{
    MatD predictions;
    unsigned long i = 0;
    for (const auto & row : X)
    {
        predictions. push_back (this->simulate (row, true)[0]);
        backpropagation (y[i]);
        updateWeight (i);
        ++i;
    }
    //Rcpp::Rcout << "average mse: " << average_loss (predictions, y) << endl;
}*/

/************** Train the network : multiple epochs *********************/
void Network::fit (const MatD & X, const MatD & y, int n_iters, bool shuffle /*= true*/)
{

    for (int i = 0; i < n_iters; ++i)
    {
        MatD X_shuffled (X);
        MatD y_shuffled (y);

        if (shuffle)
        {
            shuffle_X_y (X_shuffled, y_shuffled);
            this->train (X_shuffled, y_shuffled);
        }
        else
        {
            this->train (X, y);
        }
    }
}
/************** Train the network : multiple epochs *********************/
/*void Network::fit (const vector<MatD> & X, const MatD & y, int n_iters, bool shuffle)
{
    for (int i = 0; i < n_iters; ++i)
    {
       this->train (X, y);
    }
}*/


/********** test the network *************/
MatD Network::predict (const MatD & X)
{
    MatD predictions;
    for (const auto & row : X)
        predictions. push_back (this->simulate ({row}, false)[0]);

    return predictions;
}
/********** test the network *************/
/*MatD Network::predict(const vector<MatD> &X)
{
    MatD predictions;
    for (const auto & row : X)
        predictions. push_back (this->simulate (row, false)[0]);

    return predictions;
}*/

/********** Comput the R² score *************/
/*VectD Network::score(const vector<MatD> &X, const MatD &y)
{
    return r_score (y, predict (X));
}*/


/********** Comput the R² score *************/
VectD Network::score(const MatD & X, const MatD & y)
{
    return r_score (y, predict (X));
}

void Network::summary()
{
    unsigned i = 1;
    Rcpp::Rcout << "------------------------------------------------------\n";
    Rcpp::Rcout << "Layer_number" << std::setw(5 + 12) << "Input_dim"<< std::setw(5 + 10)  <<"Output_dim \n";

    for (auto layer: this->layers)
    {
        Rcpp::Rcout << i << std::setw (8 + 12) << layer->get_input_dim()[0] << std::setw (5 + 10) <<layer->get_output_dim()[0] << "\n";
        Rcpp::Rcout << "------------------------------------------------------" << "\n";
        ++i;
    }
}
