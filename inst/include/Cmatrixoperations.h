/**
 * @authors Hmamouche Youssef
 * @date    09/06/2016
 **/

#ifndef CMATRIXOPERATIONS_H
#define CMATRIXOPERATIONS_H

#include "Struct.h"
#include "Exception.h"

using namespace Struct;

namespace MatrixOperations {
    bool regression (const CMatDouble &, const CVDouble &,  CVDouble &); // throw (Exception);
    
    void P_Part (CVDouble &  , CMatDouble & , CMatDouble & , unsigned int)
    ;
    
    void Pr_Part (CVDouble & , CMatDouble & , unsigned int );
    
    void Diff (CVDouble & );

    CVDouble VECbivar (CMatDouble  , unsigned , bool d /* = false */); // throw (Exception);
};


#endif // CMATRIXOPERATIONS_H
