/**
 * Operations.h
 *   Purpose: Calculates matrix and vectors multiplications.
 * @authors Hmamouche Youssef
 * @date 2016 
 **/

#ifndef OPERATEURS_H
#define OPERATEURS_H

#include <Rcpp.h>

#include "Struct.h"
#include "Exception.h"

using namespace Struct;

/**
    Compute a vector-vector multiplication.

    @param A the first vector.
    @param B the second vector.
    @param Res the vector where tu put the result.
*/
void MultCVDouble (const CVDouble & A, const CVDouble & B, CVDouble & Res); 

/**
    Compute a matrix-vector multiplication.

    @param A the matrix.
    @param B the  vector.
    @param Res the vector where tu put the result.
*/
void MultCVDouble (const CMatDouble & A, const CVDouble & B, CVDouble & Res); 
;

/**
    Compute a matrix-matrix multiplication.

    @param A the first matrix.
    @param B the second matrix.
    @param Res the matrix where tu put the result.
*/
void MultCVDouble (const CMatDouble & A, const CMatDouble & B, CMatDouble & Res); 

#endif // OPERATEURS_H
