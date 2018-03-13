/**
 *
 * @file    CDynamicCausalityTest.cpp
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
#include <Rcpp.h>

#include "../inst/include/CDynamicCausality.h"
#include "../inst/include/Ftable.h"
#include "../inst/include/FDIST.h"
#include "../inst/include/CNeuralNVarModel.h"

using namespace Struct;
using namespace std;

/************************************/
DynamicCausalityTest::DynamicCausalityTest (Rcpp::NumericVector  ts1_,
                                            Rcpp::NumericVector  ts2_,
                                            int lag_,
                                            Rcpp::IntegerVector  hiddenLayersOfUnivModel,
                                            Rcpp::IntegerVector  hiddenLayersOfBivModel,
                                            int interations,
                                            bool bias) throw ()
                      {
    
    // Checking if the lag value is positif
      try {
          if (lag_ <= 0)
              throw string ("Error: The lag value is incorrect, try strictly positive value.");
      }
      catch(string const& chaine)
      {
          Rcpp::Rcout << chaine << endl;
      }
                          
    lag  = lag_;
    
    for (const auto val:ts1_)
        ts1.push_back (val);
    
    for (const auto val:ts2_)
        ts2.push_back (val);
    
    if (ts1.size() != ts2.size())
            throw string ("Error: The variables have not the same length.");
    
    // Size of hidden layers
    vector<int> sizeOfLayersModel1_ = Rcpp::as < vector<int>  > (hiddenLayersOfUnivModel);
    vector<int> sizeOfLayersModel2_ = Rcpp::as < vector<int>  > (hiddenLayersOfBivModel);
                          
    // variables
    unsigned nl (ts1.size ());
    double RSS1(0), RSS0(0);
                          
    CMatDouble M;
    M.reserve (2);
    M.push_back (ts1);
                          
    CNeuralNVarModel univariateModel (sizeOfLayersModel1_, lag, bias);
    univariateModel.fit (M, interations);
                          
    M.push_back(ts2);
                          
    CNeuralNVarModel bivariateModel (sizeOfLayersModel2_, lag, bias);
    bivariateModel.fit (M,  interations);
    
    RSS0 = univariateModel.getSSR () [0];
    RSS1 = bivariateModel.getSSR ()  [0];
                          
    int T = nl - lag;
    
    /* F test */
    Ftest = ((RSS0 - RSS1) / lag) / (RSS1 / (T - 2*lag - 1));

    /* p-value of the F-test */
    p_value = getPvalue (Ftest , lag , T - 2*lag - 1);
    
    if (lag <= 20 and T - 2*lag - 1 <= 100)
        criticTest = ftable[T - 2*lag - 2][lag];
    else if (lag > 20 and T - 2*lag - 1 <= 100)
        criticTest = ftable[T - 2*lag - 2][20];
    else if (lag <= 20 and T - 2*lag - 1 > 100)
        criticTest = ftable[100][lag];
    else if (lag > 20 and T - 2*lag - 1 > 100)
        criticTest = ftable[100][20];
    
}
void DynamicCausalityTest::summary ()
{
    Rcpp::Rcout <<  "---------------------------------------------------------\n";
    Rcpp::Rcout <<  "         Results of the non-linear Granger causality test" << "\n";
    Rcpp::Rcout <<  "---------------------------------------------------------\n";
    Rcpp::Rcout <<  "The lag parameter: p = "<< lag << "\n";
    Rcpp::Rcout <<  "The value of the F-test: "<< Ftest << "\n";
    Rcpp::Rcout <<  "The p_value of the F-test: "<< p_value << "\n";
    Rcpp::Rcout <<  "The critical value at 5% of risk: "<< criticTest <<"\n";
}
double DynamicCausalityTest::get_p_value () {
    return p_value;
}
double DynamicCausalityTest::get_F_test () {
    return Ftest;
}


