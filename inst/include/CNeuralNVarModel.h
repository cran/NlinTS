/**
 * @authors Hmamouche Youssef
 * @date    02/07/2017
 **/

#ifndef CNEURALNVARMODEL_H
#define CNEURALNVARMODEL_H

#include <cstdlib>
#include <memory>

#include "CMultiLayerPerceptron.h"

class CNeuralNVarModel
{
private:
    unsigned  _numLayers;
    std::vector<int> _sizeOfLayers;
    std::vector <std::shared_ptr <MultiLayerPerceptron> > mpl;
    std::vector<double> SSR;
    bool _bias;
    unsigned  lag;  // lag parameter
    unsigned  Nb_Ln ;    // nombre de lignes
    unsigned  Nb_Cl ;    // nombre d'attributs
    Struct::CMatDouble inputMat ;    // input data
    
public:
    CNeuralNVarModel (const std::vector<int> & ,
                      unsigned,
                      bool bias = true);

    CNeuralNVarModel (){};
    
    ~CNeuralNVarModel(){};
    
    void fit (const Struct::CMatDouble &,
              unsigned
              );
    
    Struct::CMatDouble forecast (const Struct::CMatDouble & M);
    
    void train (const Struct::CMatDouble & M);
    
    Struct::CVDouble getMSE ();
    Struct::CVDouble getMAE ();
    
    // Sum of squared errors
    std::vector<double> getSSR () {return SSR;};
    
};
#endif // CNEURALNVARMODEL_H
