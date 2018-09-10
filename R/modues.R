library (Rcpp)
library (Rdpack)

.onLoad <- function(libname, pkgname) {
}


#' Artificial Neural Network VAR (Vector Auto-Regressive) model using a MultiLayer Perceptron.
#'
#' @details This function builds the model, and returns an object that can be used to make forecasts and can be updated using new data.
#' @param df A numerical dataframe
#' @param sizeOfHLayers Integer vector that contains the size of hidden layers, where the length of this vector is the number of hidden layers, and the i-th element is the number of neurons in the i-th hidden layer.
#' @param lag The lag parameter.
#' @param iters The number of iterations.
#' @param bias Logical, true if the bias have to be used in the network.
#' @return train (df):  updates the model using the input dataframe.
#' @return forecast (df):  returns the next row forecasts of an given dataframe.
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


#' The Granger causality test
#'
#' @details The test evaluates if the second time series causes the first one using the Granger test of causality.
#' @param ts1 Numerical dataframe containing one variable.
#' @param ts2 Numerical dataframe containing one variable.
#' @param lag The lag parameter.
#' @param diff Logical argument for the option of making data stationary before making the test.
#' @return pvalue: the p-value of the test.
#' @return Ftest:  the statistic of the test.
#' @return summary ():  shows the test results.
#' @references{
#'   \insertRef{granger1980}{NlinTS}
#' }
#' @importFrom Rdpack reprompt
#' @examples
#' library (timeSeries) # to extract time series
#' library (NlinTS)
#' data = LPP2005REC
#' model = causality.test (data[,1], data[,2], 2)
#' model$summary ()
causality.test <- function(ts1,ts2, lag, diff = FALSE){
    Caus.test =  Module ('CausalityTest', PACKAGE = "NlinTS")
    test0 = Caus.test $ CausalityTest ;
    v = new (test0, ts1,ts2, lag, diff) ;
    return (v) ;
}

#' A non linear Granger causality test
#'
#' @details The test evaluates if the second time series causes the first one. Two MLP artificial neural networks are evaluated to perform the test, one using just the target time series (ts1), and the second using both time series.
#' @param ts1 Numerical series.
#' @param ts2 Numerical series.
#' @param lag The lag parameter
#' @param LayersUniv Integer vector that contains the size of hidden layers of the univariate model. The length of this vector is the number of hidden layers, and the i-th element is the number of neurons in the i-th hidden layer.
#' @param LayersBiv Integer vector that contains the size of hidden layers of the bivariate model. The length of this vector is the number of hidden layers, and the i-th element is the number of neurons in the i-th hidden layer.
#' @param iters The number of iterations.
#' @param bias Logical argument  for the option of using the bias in the networks.
#' @return pvalue: the p-value of the test.
#' @return Ftest:  the statistic of the test.
#' @return summary ():  shows the test results.
#' @examples
#' library (timeSeries) # to extract time series
#' library (NlinTS)
#' data = LPP2005REC
#' # We construct the model based
#' model = nlin_causality.test (data[,1], data[,2], 2, c(2, 2), c(4, 4), 500, TRUE)
#' model$summary ()
nlin_causality.test <- function(ts1,ts2, lag,LayersUniv, LayersBiv, iters, bias=TRUE){
    dynCaus.test =  Module ('NlinCausalityTest', PACKAGE = "NlinTS")
    test0 = dynCaus.test $ DynamicCausalityTest ;
    v = new (test0, ts1,ts2, lag, LayersUniv, LayersBiv, iters, bias) ;
    return (v) ;
}


#' Augmented Dickey_Fuller test
#'
#' @details Computes the stationarity test for a given univariate time series.
#' @param ts Numerical dataframe.
#' @param lag The lag parameter.
#' @return df: returns the value of the test.
#' @return summary ():  shows the test results.
#' @references{
#'   \insertRef{elliott1992efficient}{NlinTS}
#' }
#' @importFrom Rdpack reprompt
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



