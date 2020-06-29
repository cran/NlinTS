/*  NlinTS -- Rcpp package for non-linear time series analysis
  Copyright (C) 2015 - 2019  Hmamouche youssef
  This file is part of NlinTS
  NlinTS is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 2 of the License, or
  (at your option) any later version.
  NlinTS is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
*/

#include <Rcpp.h>

#include "../inst/include/CVARMLPExport.h"
#include "../inst/include/Tests.h"
#include "../inst/include/CausalityTest.h"
#include "../inst/include/CDynamicCausality.h"
#include "../inst/include/Entropy.h"

using namespace Rcpp ;

RCPP_MODULE (VAR_MLP) {
/* 'The VARMLP module */
  class_<CRcppExport> ("VAR_MLP")
        .constructor <Rcpp::DataFrame, unsigned, Rcpp::IntegerVector, unsigned, bool > ()
        .property("MSE", &CRcppExport::getSSR, "mean squared error of the model on training data")
        //.method ("fit", &CRcppExport::fit, "fit the model")
        .method ("train", &CRcppExport::train, "Update the model from input data")
        .method ("forecast", &CRcppExport::forecast, "Computes the predictions")
    ;
}
/* 'The dickey fuller test module. */
RCPP_MODULE (DickeyFuller) {
  class_<DickeyFuller> ("DickeyFuller")
        .constructor <Rcpp::NumericVector, int> ()
        .method ("summary", &DickeyFuller::summary, "Summary of the test")
        .property ("df", &DickeyFuller::getDF, "return the value of test");
}

/* 'The Granger causality test module. */
RCPP_MODULE (CausalityTest) {
  class_<CausalityTest> ("CausalityTest")
        .constructor <Rcpp::NumericVector, Rcpp::NumericVector, int, bool> ()
        .method ("summary", &CausalityTest::summary, "Summary of the test")
        .property ("pvalue", &CausalityTest::get_p_value, "return the p-value of the test")
        .property ("gci", &CausalityTest::get_gci, "return the granger causality index of the test")
        .property ("Ftest", &CausalityTest::get_F_test, "return the value of F test");
}
 /*'The non linear Granger CausalityTest module. */
 RCPP_MODULE (NlinCausalityTest) {
    class_<DynamicCausalityTest> ("DynamicCausalityTest")
        .constructor <  Rcpp::NumericVector,
                        Rcpp::NumericVector,
                        int,
                        Rcpp::IntegerVector,
                        Rcpp::IntegerVector,
                        int,
                        bool> ()
    .method ("summary", &DynamicCausalityTest::summary, "Summary of the test")
    .property ("gci", &DynamicCausalityTest::get_gci, "return the granger causality index of the test")
    .property ("pvalue", &DynamicCausalityTest::get_p_value, "returns the p-value of the test")
    .property ("Ftest", &DynamicCausalityTest::get_F_test, "returns the value of F test");
}

RCPP_MODULE (InfoEntropy) {
  function( "entropy_disc", &entropy_disc);
  function( "mutualInformation_disc", &mutualInformation_disc);
  function( "mutualInformation_disc_u", &mutualInformation_disc_u);
  function( "transferEntropy_disc", &transferEntropy_disc);

  function( "entropy_cont", &entropy_cont);
  function( "mutualInformation_cont", &mutualInformation_cont);
  function( "transferEntropy_cont", &transferEntropy_cont);
}
