/**
 * @authors Hmamouche Youssef
 **/

#include "../inst/include/CNeuralNVarModel.h"
#include "../inst/include/Cmatrixoperations.h"


using namespace MatrixOperations;
using namespace Struct;
using namespace std;

/****************************************************************/
CNeuralNVarModel::CNeuralNVarModel(const vector<int> & sizeOfLayers,
                                   unsigned p,
                                   bool bias):
_numLayers (sizeOfLayers.size () + 2),
_sizeOfLayers (sizeOfLayers),
_bias (bias),
lag (p)
{
    mpl.clear ();
}

/****************************************************************/
void CNeuralNVarModel::fit (const Struct::CMatDouble & M,
                            unsigned iterations)
{
    unsigned int i, c;
    double mse;
    CMatDouble A, Prediction, Real;
    CVDouble result, one, res_out;
    CMatDouble minMax;
    
    // Resize the neural networks, one network for each target variable
    mpl.clear ();
    mpl.resize (M.size ());
    
    // Training matrix
    inputMat = M;
    
    // Scale the input data in [0,1]
    minMax = inputMat.Normalise ();
    
    // Dimensions
    Nb_Ln = M[0].size ();
    Nb_Cl = M.size ();
    
    // Construct the training matrix
    for (auto & vec:inputMat)
        P_Part (vec, Real, A, lag);

    
    // Build the networks:
    _sizeOfLayers.insert (_sizeOfLayers.begin(),  A.size ());
    _sizeOfLayers.push_back (1);
    
    // Train the networks
    c = 0;
    for (const auto &vec:Real){
        mpl[c] = shared_ptr <MultiLayerPerceptron> (new MultiLayerPerceptron (_numLayers, _sizeOfLayers, _bias));
        mpl[c]->Run (vec, A, iterations, res_out);
        ++c;
    }
    
    // make forecasts of training data
    for (auto & network:mpl)
    {
        network->Test (A, result);
        Prediction.push_back(result);
        result.clear();
    }
    
    // Model accuracy
    SSR.clear ();
    SSR.resize (Real.size ());
    
    for (i = 0 ; i < Real.size () ; ++i)
    {
        mse = 0;
        for (unsigned m = 0 ; m < Real[i].size() ; m++)
            mse += pow (Prediction[i][m] - Real[i][m], 2);
        
        SSR [i] = mse;
    }
}
/******************* the Simulation **********************/
CMatDouble CNeuralNVarModel::forecast (const Struct::CMatDouble & M)
{
     CMatDouble F, A, B, Res, present;
    CVDouble result, one;
    
    Res = M;
    
    CMatDouble minMax;
 
    
    minMax = Res.Normalise ();
    
    if (Res.size () == Nb_Cl and Res[0].size () >= lag)
    {
        for (auto vec:Res)
            Pr_Part (vec, A, lag);
        
        Res.clear ();
        
        // Simulate the networks for each time series
        for (auto & network:mpl)
        {
            network->Test (A, result);
            Res.push_back(result);
            result.clear();
        }
    }

    Res.Denormalising (minMax);
    
    return Res;
    
}

/******************* the Simulation **********************/
void CNeuralNVarModel::train (const Struct::CMatDouble & M)
{
    CMatDouble B, present, Res, minMax;
    CVDouble result, one;
    
    Res = M;
    
    minMax = Res.Normalise();
    
    if (Res.size () == Nb_Cl and Res[0].size () >= lag)
    {
        for (auto vec:Res)
            P_Part (vec, present, B, lag);
        
        // Simulate the networks for each time series
        int c = 0;
        for (const auto &vec:present)
        {
            mpl[c]->Train (vec, B);
            c++;
        }
    }
}

/*************** forecast accuracy measures **************/

CVDouble CNeuralNVarModel::getMSE ()
{
    CVDouble res;
    
    for (auto & network:mpl)
        res.push_back (network->getMSE ());
                       
    return res;
}
                       
CVDouble CNeuralNVarModel::getMAE ()
{
    CVDouble res;
    
    for (auto & network:mpl)
        res.push_back (network->getMAE ());
                       
    return res;
}



