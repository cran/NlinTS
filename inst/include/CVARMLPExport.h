/*!
 * @file CRcppExport.h
 *
 * @author youssef hmamouche
 *
 * @date 15/06/2015
 *
 * @brief Main classe of the module VARMLP, allowing the interaction with C++ class.
 *
 */

#pragma once


#include <Rcpp.h>

#include "CNeuralNVarModel.h"

class CRcppExport
{
private:

    Struct::CMatDouble M;
    CNeuralNVarModel Obj;

public:

    CRcppExport (Rcpp::DataFrame Df, unsigned p, Rcpp::IntegerVector, unsigned iters, bool bias);
    ~CRcppExport(){};
    Rcpp::DataFrame forecast (Rcpp::DataFrame DF);
    void fit (Rcpp::DataFrame, unsigned);
    void train (Rcpp::DataFrame DF);
    Rcpp::NumericVector getSSR ();
};

