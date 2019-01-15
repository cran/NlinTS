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
    double GCI;
    double criticTest;
    
public:
    
    /**
    @param ts1_ the first univariate time series as a vector.
    @param ts2_ the second time series.
    @param lag_ the lag parameter.
    @param hiddenLayersOfUnivModel vector of hidden layers sizes for the univariate model.
    @param hiddenLayersOfUnivModel vector of hidden layers sizes for the bivariate model.
    @param d a boolean value for the possibility of making data stationarr in case of true.
    @param iterations the number of iterations.
    @param d a boolean value for the possibility of making data stationarr in case of true.
    */

    DynamicCausalityTest (Rcpp::NumericVector  ts1_,
                          Rcpp::NumericVector  ts2_,
                          int lag_,
                          Rcpp::IntegerVector  hiddenLayersOfUnivModel,
                          Rcpp::IntegerVector  hiddenLayersOfBivModel,
                          int iterations,
                          bool bias = true
                          );

    DynamicCausalityTest (){};
    ~DynamicCausalityTest (){};
    
    // The causality index
    double get_gci ();

    // Get the p-value of the test
    double get_p_value ();

    // Get the  statistic of the test
    double get_F_test () ;

    // The Summary function
    void summary ();
    };
#endif
