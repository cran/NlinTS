/*
 MLP Library - Version 2.0 - August 2005
 Copyright (c) 2005 Sylvain BARTHELEMY
 Contact: sylbarth@gmail.com, www.sylbarth.com
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the "Software"),
 to deal in the Software without restriction, including without limitation
 the rights to use, copy, modify, merge, publish, distribute, sublicense,
 and/or sell copies of the Software, and to permit persons to whom the
 Software is furnished to do so, subject to the following conditions:
 The above copyright notice and this permission notice shall be included
 in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
 OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

/* Update
 * Add bias neurons to each hidden layer and change the input structure.
 * Adapt code to work with Rcpp
 * by: Youssef Hmamouche 2017-2018
 */

#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <ctime>
#include <random>
#include <Rcpp.h>
#include <random>

#include "../inst/include/CMultiLayerPerceptron.h"

using namespace Struct;
using namespace std;


void InitializeRandoms()
{
    mt19937 mt_rand(time(0));
}

double RandomEqualREAL()
{
    return Rcpp::runif(1)[0];
}

MultiLayerPerceptron::MultiLayerPerceptron(int nl,  const vector<int> &np, bool bias) :
nNumLayers(0),
pLayers(0),
dMSE(0.0),
dMAE(0.0),
dEta(0.25),
//dEta(0.8),
dAlpha(0.9),
dGain(1.0),
dAvgTestError(0.0)
{
    int i,j;
    vector<int> npl (np);
    // check if  bias neurones have to be used
    if (bias)
    {
        ind_bias = 1;
        for (i = 0; i < npl.size (); ++i)
            npl[i] += 1;
    }
    else ind_bias = 0;

    
    /* --- creation of layers */
    nNumLayers = nl;
    pLayers    = new Layer[nl];
    
    /* --- init des couches */
    for ( i = 0; i < nNumLayers; i++ )
    {
        /* --- creation of neurons */
        pLayers[i].nNumNeurons = npl[i];
        pLayers[i].pNeurons    = new Neuron[ npl[i] ];
        
        /* --- init of neurons */
        for( j = 0; j < npl[i]; j++ )
        {
                pLayers[i].pNeurons[j].x  = 1.0;
                
                pLayers[i].pNeurons[j].e  = 0.0;

            if (i > 0 && j >= ind_bias)
            {
                pLayers[i].pNeurons[j].w     = new double[ npl[i-1] ];
                pLayers[i].pNeurons[j].dw    = new double[ npl[i-1] ];
                pLayers[i].pNeurons[j].wsave = new double[ npl[i-1] ];
            }
            else
            {
                pLayers[i].pNeurons[j].w     = NULL;
                pLayers[i].pNeurons[j].dw    = NULL;
                pLayers[i].pNeurons[j].wsave = NULL;
            }
        }
        
    }
}

/**************************************************/
MultiLayerPerceptron::~MultiLayerPerceptron()
{
    int i,j;
    for( i = 0; i < nNumLayers; i++ )
    {
        if ( pLayers[i].pNeurons )
        {
            for( j = 0; j < pLayers[i].nNumNeurons; j++ )
            {
                if ( pLayers[i].pNeurons[j].w )
                    delete[] pLayers[i].pNeurons[j].w;
                if ( pLayers[i].pNeurons[j].dw )
                    delete[] pLayers[i].pNeurons[j].dw;
                if ( pLayers[i].pNeurons[j].wsave )
                    delete[] pLayers[i].pNeurons[j].wsave;
            }
        }
        delete[] pLayers[i].pNeurons;
    }
    delete[] pLayers;
}

/**************************************************/
void MultiLayerPerceptron::RandomWeights()
{
    int i,j,k;
    for( i = 1; i < nNumLayers; i++ )
    {
        for( j = ind_bias; j < pLayers[i].nNumNeurons; j++ )
        {
            for ( k = 0; k < pLayers[i-1].nNumNeurons; k++ )
            {
                pLayers[i].pNeurons[j].w [k]    = RandomEqualREAL();
                pLayers[i].pNeurons[j].dw[k]    = 0.0;
                pLayers[i].pNeurons[j].wsave[k] = 0.0;
            }
        }
    }
}

/**************************************************/
void MultiLayerPerceptron::SetInputSignal(CVDouble & input)
{
    int i;
    
    for ( i = ind_bias; i < pLayers[0].nNumNeurons; i++)
    {
        pLayers[0].pNeurons[i].x = input[i-ind_bias];
    }
    if (ind_bias == 1)
        pLayers[0].pNeurons[0].x = 1;
    
    
}

/**************************************************/
void MultiLayerPerceptron::GetOutputSignal(CVDouble & output)
{
    int i;
    output.clear ();
    
    output.resize (pLayers[nNumLayers-1].nNumNeurons - ind_bias);
    
    for ( i = ind_bias; i < pLayers[nNumLayers-1].nNumNeurons; i++ )
        output[i-ind_bias] = pLayers[nNumLayers-1].pNeurons[i].x;
}

/**************************************************/
void MultiLayerPerceptron::SaveWeights()
{
    int i,j,k;
    for( i = 1; i < nNumLayers; i++ )
        for( j = ind_bias; j < pLayers[i].nNumNeurons; j++ )
            for ( k = 0; k < pLayers[i-1].nNumNeurons; k++ )
                pLayers[i].pNeurons[j].wsave[k] = pLayers[i].pNeurons[j].w[k];
}

/**************************************************/
void MultiLayerPerceptron::RestoreWeights()
{
    int i,j,k;
    for( i = 1; i < nNumLayers; i++ )
        for( j = ind_bias; j < pLayers[i].nNumNeurons; j++ )
            for ( k = 0; k < pLayers[i-1].nNumNeurons; k++ )
                pLayers[i].pNeurons[j].w[k] = pLayers[i].pNeurons[j].wsave[k];
}

/***************************************************************************/
/* calculate and feedforward outputs from the first layer to the last      */
void MultiLayerPerceptron::PropagateSignal()
{
    int i,j,k;
    
    /* --- la boucle commence avec la seconde couche */
    for (i = 1; i < nNumLayers; i++)
    {
        for (j = ind_bias; j < pLayers[i].nNumNeurons; j++)
        {
            /* --- calcul de la somme ponderee en entree */
            double sum = 0.0;
            for (k = 0 ; k < pLayers[i-1].nNumNeurons ; ++k)
            {
                double out = pLayers[i-1].pNeurons[k].x;
                double w   = pLayers[i  ].pNeurons[j].w[k];
                sum       += w * out;
            }
            /* --- application de la fonction d'activation (sigmoid) */
            pLayers[i].pNeurons[j].x = 1.0 / (1.0 + exp(-dGain * sum));
        }
    }
}

/**************************************************/
void MultiLayerPerceptron::ComputeOutputError(CVDouble & target)
{
    int  i;
    dMSE = 0.0;
    dMAE = 0.0;
    for( i = ind_bias; i < pLayers[nNumLayers-1].nNumNeurons; i++)
    {
        double x = pLayers[nNumLayers-1].pNeurons[i].x;
        double d = (target[i- ind_bias] - x);
        
        pLayers[nNumLayers-1].pNeurons[i].e = dGain * x * (1.0 - x) * d;
        
        dMSE += (d * d);
        dMAE += fabs(d);
    }
    /* --- erreur quadratique moyenne */
    dMSE /= (double)(pLayers[nNumLayers-1].nNumNeurons - ind_bias);
    
    /* --- erreur absolue moyenne */
    dMAE /= (double)(pLayers[nNumLayers-1].nNumNeurons - ind_bias);
}

/***************************************************************************/
/* backpropagate error from the output layer through to the first layer    */

void MultiLayerPerceptron::BackPropagateError()
{
    int i,j,k;
    /* --- la boucle commence a l'avant derniere couche */
    for( i = (nNumLayers-2); i >= 0; i-- )
    {
        /* --- couche inferieure */
        for( j = ind_bias; j < pLayers[i].nNumNeurons; j++ )
        {
            double x = pLayers[i].pNeurons[j].x;
            double E = 0.0;
            /* --- couche superieure */
            for ( k = ind_bias; k < pLayers[i+1].nNumNeurons; k++ )
            {
                E += pLayers[i+1].pNeurons[k].w[j] * pLayers[i+1].pNeurons[k].e;
            }
            pLayers[i].pNeurons[j].e = dGain * x * (1.0 - x) * E;
        }
    }
}

/***************************************************************************/
/* update weights for all of the neurons from the first to the last layer  */

void MultiLayerPerceptron::AdjustWeights()
{
    int i,j,k;
    /* --- la boucle commence avec la seconde couche */
    for( i = 1; i < nNumLayers; i++ )
    {
        for( j = ind_bias; j < pLayers[i].nNumNeurons; j++ )
        {
            for ( k = 0; k < pLayers[i-1].nNumNeurons; k++ )
            {
                double x  = pLayers[i-1].pNeurons[k].x;
                double e  = pLayers[i  ].pNeurons[j].e;
                double dw = pLayers[i  ].pNeurons[j].dw[k];
                pLayers[i].pNeurons[j].w [k] += dEta * x * e + dAlpha * dw;
                pLayers[i].pNeurons[j].dw[k]  = dEta * x * e;
            }
        }
    }
}

/**************************************************/
void MultiLayerPerceptron::Simulate(CVDouble & input, CVDouble & output, CVDouble & target, bool training)
{
    
    if(0 == input.size())  return;
    if(0 == target.size()) return;
    
    
    /* --- on fait passer le signal dans le reseau */
    
    SetInputSignal(input);
    
    PropagateSignal();
    
    //AdjustWeights();
    
    GetOutputSignal(output);
    
    
    /* --- calcul de l'erreur en sortie par rapport a la cible */
    /*     ce calcul sert de base pour la rÈtropropagation     */
    // ComputeOutputError(target);
    
    /* --- si c'est un apprentissage, on fait une retropropagation de l'erreur */
    if (training)
    {
        ComputeOutputError(target);
        BackPropagateError();
        AdjustWeights();
        //GetOutputSignal(output);
    }
    //std::cout << "dMSE: "<< pLayers[nNumLayers-1].pNeurons[0].x << std::endl;
    
}

/**************************************************/
void MultiLayerPerceptron::Predict (CVDouble & input, Struct::CVDouble & output)
{
    if(0 == input.size())  return;
    
    SetInputSignal(input);
    PropagateSignal();
    GetOutputSignal(output);
}

/**************************************************/

int MultiLayerPerceptron::Train(const CVDouble & target_, const CMatDouble & trainingMatrix)
{
    int count = 0;
    
    unsigned int Numb = trainingMatrix[0].size ();
    unsigned int ncol = trainingMatrix.size ();
    
    CVDouble  input (pLayers[0].nNumNeurons);
    CVDouble  output (pLayers[nNumLayers-1].nNumNeurons);
    CVDouble  target (pLayers[nNumLayers-1].nNumNeurons);
    
    if(0 == input.size()) return 0;
    if(0 == output.size()) return 0;
    if(0 == target.size()) return 0;
    
    for (unsigned int i = 0 ; i < Numb ; ++i)
    {
        /* --- on le transforme en input/target */
        target[0] = target_[i];
        for (unsigned int j = 0 ; j < ncol ; ++j)
            input[j] = trainingMatrix[j][i];
        
        /* --- on fait un apprentisage du reseau avec cette ligne*/
        Simulate(input, output, target, true);
        count++;
    }
    
    input.clear();
    output.clear();
    target.clear();
    
    return count;
}

/*******************************************************/
int MultiLayerPerceptron::Test(const CMatDouble & Matrix, CVDouble & Output)
{
    int count = 0;
    unsigned int Numb = Matrix[0].size ();
    unsigned int ncol = Matrix.size ();
    Output.resize (Numb);
    
    CVDouble  input (pLayers[0].nNumNeurons);
    CVDouble  output (pLayers[nNumLayers-1].nNumNeurons);
    
    if(0 == Matrix.size() or 0 == Matrix[0].size()) return 0;
    if(0 == input.size()) return 0;
    if(0 == output.size()) return 0;
    
    dAvgTestError = 0.0;
    
    CMatDouble trainingMatrix = Matrix;
    
    for (unsigned int i = 0 ; i < Numb ; ++i)
    {
        for (unsigned int j = 0 ; j < ncol ; ++j)
            input[j] = trainingMatrix[j][i];
        
        /* --- on fait un apprentisage du reseau  avec cette ligne*/
        Predict(input,output);
        Output[i] = output[0];
        count++;
    }
    
    dAvgTestError /= count;
    
    input.clear();
    output.clear();
    
    return count;
}

int MultiLayerPerceptron::Evaluate()
{
    int count = 0;
    return count;
}

void MultiLayerPerceptron::Run(const CVDouble & target_, const CMatDouble & Matrix, const int& maxiter, CVDouble & Output)
{
    int    countTrain = 0;
    int    countLines = 0;
    bool   Stop = false;
    bool   firstIter = true;
    double dMinTestError = 0.0;
    
    CMatDouble trainingMatrix = Matrix;
    CVDouble cible = target_;
    
    Output.clear ();
    
    /* --- init du generateur de nombres aleatoires  */
    /* --- et generation des pondÈrations aleatoires */
    InitializeRandoms();
    RandomWeights();
    
    /* --- on lance l'apprentissage avec tests */
    do {
        
        countLines += Train(target_, trainingMatrix);
        //Test(trainingMatrix);
        countTrain++;
        
        if(firstIter)
        {
            dMinTestError = dAvgTestError;
            firstIter = false;
        }
        
        // printf( "%i \t TestError: %f", countTrain, dAvgTestError);
        
        if ( dAvgTestError < dMinTestError)
        {
            //printf(" -> saving weights\n");
            dMinTestError = dAvgTestError;
            SaveWeights();
        }
        else if (dAvgTestError > 1.2 * dMinTestError)
        {
            //printf(" -> stopping training and restoring weights\n");
            //Stop = true;
            //RestoreWeights();
        }
        
    } while ( (!Stop) && (countTrain<maxiter) );
    
    //Test(trainingMatrix, Output);
    
    CVDouble input;// (trainingMatrix.size () - 1);
    
    CVDouble Res;
    
    for (unsigned i = 0 ; i < trainingMatrix[0].size (); ++i){
        for (unsigned j = 0 ; j < trainingMatrix.size (); ++j)
            input.push_back (trainingMatrix [j][i]);
        Predict (input, Res);
        Output.push_back(Res[0]);
        input.clear();
    }
}
