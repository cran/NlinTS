/**
  @file    CausalityTest.h
  Purpose: class for the Granger causality test.
  @authors Hmamouche Youssef
  @date    05/07/2017
 **/

#ifndef CAUSALITYTEST_H
#define CAUSALITYTEST_H

#include "Struct.h"
#include "Exception.h"

class CausalityTest {
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
    Construct the model

    @param ts1_ the first univariate time series as a vector.
    @param ts2_ the second time series.
    @param lag_ the lag parameter.
    @param d a boolean value for the possibility of making data stationarr in case of true.
    */
    CausalityTest (Rcpp::NumericVector ts1_,
                   Rcpp::NumericVector ts2_,
                   int lag_,
                   bool d = false); // throw (Exception);

    ~CausalityTest (){};
    
    /* The bivariate VAR model */
    friend Struct::CVDouble VECbivar (Struct::CMatDouble, unsigned, bool d);

    // The causality Index
    double get_gci ();

    // Get the p-value of the test
    double get_p_value ();

    // Get the  statistic of the test
    double get_F_test () ;

    // The Summary function
    void summary ();
    };

#endif
