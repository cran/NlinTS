/**
 * @authors Hmamouche Youssef
 * @date    05/07/2017
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
    double criticTest;
    
public:
    
    CausalityTest (Rcpp::NumericVector,
                   Rcpp::NumericVector,
                   int,
                   bool d = false); // throw (Exception);
    ~CausalityTest (){};
    
    friend Struct::CVDouble VECbivar (Struct::CMatDouble, unsigned, bool d);
    double get_p_value ();
    double get_F_test () ;
    void summary ();
    };

#endif
