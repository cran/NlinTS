library (Rcpp)

.onLoad <- function(libname, pkgname) {
}


#' VARMLP model
#'
#' @details This function construct the VAR-MLP model (vector autoregressive based on a multilayer perceptron network).
#' @param df A numerical dataframe
#' @param sizeOfHLayers Integer vector that contains the size of hidden layers (the number of hidden layers is the size of this vector).
#' @param lag The lag parameter
#' @param iters The number of iterations
#' @param bias Logical, true if the bias have to be used in the network
#' @return train (df):  update the model based on input dataframe
#' @return forecast (df):  forecast next row of an input dataframe
#' @examples
#' library (timeSeries) # to extract time series
#' library (NlinTS)
#' #load data
#' data = LPP2005REC
#' # Prepare data to make one forecasts
#' train_data = head (data, nrow (data) - 1)
#' test_data = tail (data, 1)
#' model = varmlp (train_data, 1, c(10,5), 200, TRUE)
#' predictions = model$forecast (train_data)
#' print (tail (predictions,1))
#' # Update the model (learning from new data)
#' model$train (test_data)
varmlp <- function(df, lag, sizeOfHLayers, iters, bias = TRUE){
    neuralNet =  Module ('VAR_MLP', PACKAGE = "NlinTS") ;
    varp = neuralNet$VAR_MLP
    v = new (varp, df, lag, sizeOfHLayers, iters, bias)
    return (v)
}

#' Augmented Dickey_Fuller test
#'
#' @details Computes the stationarity test for a given univariate time series
#' @param ts Numerical dataframe
#' @param lag The lag parameter
#' @return summary ():  shows the test results
#' @return df (): returns the value of the test
#' @examples
#' library (timeSeries)
#' library (NlinTS)
#' #load data
#' data = LPP2005REC
#' model = df.test (data[,1], 1)
#' model$summary ()
df.test <- function(ts, lag){
  DF.test =  Module ('DickeyFuller', PACKAGE = "NlinTS") ;
  test = DF.test $ DickeyFuller ;
  v = new(test, ts, lag) ;
  return (v) ;
}

#' The Granger causality test
#'
#' @details The test evaluates if the second time series causes the first one using the Granger test of causality
#' @param ts1 Numerical dataframe containing one variable
#' @param ts2 Numerical dataframe containing one variable
#' @param lag The lag parameter
#' @param diff Logical argument for the option of making data stationary
#' @return summary ():  shows the test results
#' @return F-test (): returns the value of the test
causality.test <- function(ts1,ts2, lag, diff = FALSE){
    Caus.test =  Module ('CausalityTest', PACKAGE = "NlinTS")
    test0 = Caus.test $ CausalityTest ;
    v = new (test0, ts1,ts2, lag, diff) ;
    return (v) ;
}

#' A non linear Granger causality test
#'
#' @details The test evaluates if the second time series causes the first one using a non-linear approach. The test evaluates two artificial neural network busing VARMLP model.
#' @param ts1 Numerical series
#' @param ts2 Numerical series
#' @param lag The lag parameter
#' @param LayersUniv Integer vector of the size of hidden layers of the univariate model
#' @param LayersBiv Integer vector of the size of hidden layers of the bivariate model
#' @param iters The number of iterations
#' @param bias Logical argument  for the option of using the bias in the networks
#' @return pvalue: the p-value of the test
#' @return Ftest:  the statistic of the test
#' @return summary ():  shows the test results
#' @return F-test (): returns the value of the test
#' @examples
#' library (timeSeries) # to extract time series
#' library (NlinTS)
#' data = LPP2005REC
#' model = nlin_causality.test (data[,1], data[,2], 2, c(2), c(4), 500, TRUE)
#' model$summary ()
nlin_causality.test <- function(ts1,ts2, lag,LayersUniv, LayersBiv, iters, bias=TRUE){
    dynCaus.test =  Module ('NlinCausalityTest', PACKAGE = "NlinTS")
    test0 = dynCaus.test $ DynamicCausalityTest ;
    v = new (test0, ts1,ts2, lag, LayersUniv, LayersBiv, iters, bias) ;
    return (v) ;
}


