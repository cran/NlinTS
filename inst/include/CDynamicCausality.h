/**
 *
 * @file    CDynamicCausalityTest.h
 *
 * @authors Hmamouche Youssef
 *
 * @date    02/07/2017
 *
 * @version V1.0
 *
 * @brief  classe CDynamicCausalityTest
 *
 **/

#ifndef CDYNAMICCAUSALITY_H
#define CDYNAMICCAUSALITY_H

#include <Rcpp.h>
#include "Struct.h"


/*****************  Dynamic causality class  *********************/

class DynamicCausalityTest {
private:
    
    Struct::CVDouble ts1;
    Struct::CVDouble ts2;
    double Ftest;
    unsigned lag;
    double p_value;
    double criticTest;
    
public:
    
    DynamicCausalityTest (Rcpp::NumericVector  ts1_,
                          Rcpp::NumericVector  ts2_,
                          int lag_,
                          Rcpp::IntegerVector  hiddenLayersOfUnivModel,
                          Rcpp::IntegerVector  hiddenLayersOfBivModel,
                          int interations,
                          bool bias = true
                          ) throw ();
    DynamicCausalityTest (){};
    ~DynamicCausalityTest (){};
    
    double get_p_value ();
    double get_F_test () ;
    void summary ();
    };
#endif
