/**
 * @authors Hmamouche Youssef
 **/

#include <Rcpp.h>
#include "../inst/include/Tests.h"
#include "../inst/include/Ftable.h"
#include "../inst/include/FDIST.h"
#include "../inst/include/Cmatrixoperations.h"
#include "../inst/include/Operateurs.h"
#include "../inst/include/CausalityTest.h"


using namespace MatrixOperations;
using namespace Struct;
using namespace std;


/************************************/
CausalityTest::CausalityTest (Rcpp::NumericVector  ts1_,
                              Rcpp::NumericVector  ts2_, int lag_, bool d /* = false */) throw (Exception) {
    
    // Checking if the lag value is positif
    if (lag_ <= 0)
        throw Exception ("The lag value is incorrect");

    lag  = lag_;
    
    for (const auto val:ts1_)
        ts1.push_back (val);
    
    for (const auto val:ts2_)
        ts2.push_back (val);
    
    if (ts1.size() != ts2.size())
       throw Exception ("Time series have not the same size");
 
    
    // variables
    unsigned nl (ts1.size ());
    double RSS1(0), RSS0(0);
    
    CVDouble VECT;
    CMatDouble M (1);
    M[0] = ts1;

    /* the VAR model of the system (v1) */
    VECT = VECbivar(M, lag, d);
    RSS0 = VECT[0];
    
    /* the VAR model of the system (v1,v2) */
    M.push_back(ts2);
    VECT = VECbivar(M, lag, d);
    RSS1 = VECT[0];
    
    int T = nl - lag;
    // F test
    Ftest = ((RSS0 - RSS1) / lag) / (RSS1 / (T - 2*lag - 1));
    
    // p-value of the F-test
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
void CausalityTest::summary ()
{
    Rcpp::Rcout <<  "------------------------------------------------\n";
    Rcpp::Rcout <<  "        Test of causality" << "\n";
    Rcpp::Rcout <<  "------------------------------------------------\n";
    Rcpp::Rcout <<  "The lag parameter: p = "<< lag << "\n";
    Rcpp::Rcout <<  "The value of the F-test: "<< Ftest << "\n";
    Rcpp::Rcout <<  "The p_value of the F-test: "<< p_value << "\n";
    Rcpp::Rcout <<  "The critical value with 5% of risk:: "<< criticTest <<"\n";
}
double CausalityTest::get_p_value () {
    return p_value;
}
double CausalityTest::get_F_test () {
    return Ftest;
}

