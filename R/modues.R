library (Rcpp)
library (Rdpack)



.Info <-  Module ('InfoEntropy', PACKAGE = "NlinTS")

.neuralNet <-  Module ('VAR_MLP', PACKAGE = "NlinTS")

.Caus.test <-  Module ('CausalityTest', PACKAGE = "NlinTS")

.dynCaus.test <-  Module ('NlinCausalityTest', PACKAGE = "NlinTS")

 
.df.test <-  Module ('DickeyFuller', PACKAGE = "NlinTS")

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
    #neuralNet =  Module ('VAR_MLP', PACKAGE = "NlinTS")
    varp = .neuralNet$VAR_MLP
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
#' @return gci: the Granger causality index.
#' @return Ftest:  the statistic of the test.
#' @return pvalue: the p-value of the test.
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
    #Caus.test =  Module ('CausalityTest', PACKAGE = "NlinTS")
    test0 = .Caus.test $ CausalityTest ;
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
#' @return gci: the Granger causality index.
#' @return Ftest:  the statistic of the test.
#' @return pvalue: the p-value of the test.
#' @return summary ():  shows the test results.
#' @examples
#' library (timeSeries) # to extract time series
#' library (NlinTS)
#' data = LPP2005REC
#' # We construct the model based
#' model = nlin_causality.test (data[,1], data[,2], 2, c(2, 2), c(4, 4), 500, TRUE)
#' model$summary ()

nlin_causality.test <- function(ts1,ts2, lag,LayersUniv, LayersBiv, iters, bias=TRUE)
{
    #dynCaus.test =  Module ('NlinCausalityTest', PACKAGE = "NlinTS") 
    test0 = .dynCaus.test $ DynamicCausalityTest ;
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
    test = .df.test $ DickeyFuller ;
    v = new(test, ts, lag) ;
    return (v) ;
}


#' Discrete Entropy
#'
#' @details Computes the Shanon entropy of an integer vector.
#' @param V Integer vector.
#' @param log String argument in the set ("log2", "loge","log10"), which indicates the log function to use. The log2 is used by default.
#' @importFrom Rdpack reprompt
#' @examples
#' library (NlinTS)
#' print (entropy_disc (c(3,2,4,4,3)))

entropy_disc <- function (V, log = "log2")
{
    v = .Info $ entropy_disc (V, log);
    return (v)
}

#*****************************************#
#' Discrete bivariate Mutual Information
#'
#' @details Computes the Mutual Information between two integer vectors.
#' @param X Integer vector.
#' @param Y Integer vector.
#' @param log String argument in the set ("log2", "loge","log10"), which indicates the log function to use. The log2 is used by default.
#' @examples
#' library (NlinTS)
#' mi = mi_disc_bi (c(3,2,4,4,3), c(1,4,4,3,3))
#' print (mi)
mi_disc_bi <- function (X, Y, log = "log2")
{
    #Info =  Module ('InfoEntropy', PACKAGE = "NlinTS")
    v = .Info $ mutualInformation_disc_u (X, Y, log);
    return (v);
}

#' Discrete multivariate Mutual Information
#'
#' @details Computes the Mutual Information between columns of a dataframe.
#' @param df  Datafame of type Integer.
#' @param log String argument in the set ("log2", "loge","log10"), which indicates the log function to use. The log2 is used by default.
#' @examples
#' library (NlinTS)
#' df = data.frame (c(3,2,4,4,3), c(1,4,4,3,3))
#' mi = mi_disc (df)
#' print (mi)
mi_disc <- function (df, log = "log2")
{
    #Info =  Module ('InfoEntropy', PACKAGE = "NlinTS")
    M = t(df)
    v = .Info $ mutualInformation_disc (M, log);
    return (v);
}

#*****************************************#
#' Discrete  Transfer Entropy
#'
#' @details Computes the Transfer Entropy from the second time series to the one.
#' @param X Integer vector, first time series.
#' @param Y Integer vector, the second time series.
#' @param p Integer, the lag parameter to use for the first vector (p = 1 by default).
#' @param q Integer, the lag parameter to use for the first vector (q = 1 by default)..
#' @param log String argument in the set ("log2", "loge","log10"), which indicates the log function to use. The log2 is used by default.
#' @param normalize Logical argument  for the option of normalizing the value of TE (transfer entropy) (FALSE by default).
#' This normalization is done by deviding TE by H (X(t)| X(t-1), ..., X(t-p)), where H is the Shanon entropy.
#' @references{
#'   \insertRef{schreiber2000}{NlinTS}
#' }
#' @importFrom Rdpack reprompt
#' @examples
#' library (NlinTS)
#' te = te_disc (c(3,2,4,4,3), c(1,4,4,3,3), 1, 1)
#' print (te)
te_disc <- function (X, Y, p = 1, q = 1, log = "log2", normalize = FALSE)
{
    #Info =  Module ('InfoEntropy', PACKAGE = "NlinTS")
    v = .Info $ transferEntropy_disc (X, Y, p, q, log, normalize);
    return (v);
}

#' Continuous  entropy 
#'
#' @details Computes the continuous entropy of a numerical vector using the Kozachenko approximation.
#' @param V  Interger vector.
#' @param k  Integer argument, the number of neighbours.
#' @references{
#'   \insertRef{kraskov2004estimating}{NlinTS}
#' }
#' @importFrom Rdpack reprompt
#' @examples
#' library (timeSeries)
#' library (NlinTS)
#' #load data
#' data = LPP2005REC
#' print (entropy_cont (data[,1], 3))
 entropy_cont <- function (V, k = 3)
 {
    #Info =  Module ('InfoEntropy', PACKAGE = "NlinTS")
     v = .Info $ entropy_cont (V, k);
     return (v)
 }

# #*****************************************#
#' Continuous  Mutual Information
#'
#' @details Computes the   Mutual Information between columns of a dataframe using the Kraskov approximation.
#' @param df  Datafame of type Integer
#' @param k  Integer argument, the number of neighbours.
#' @param algo String argument specifies the algorithm use ("ksg1", "ksg2"), as tow propositions of Kraskov approximation are provided. The first one ("ksg1") is used by default
#' @references{
#'   \insertRef{kraskov2004estimating}{NlinTS}
#' }
#' @importFrom Rdpack reprompt
#' @examples
#' library (timeSeries)
#' library (NlinTS)
#' #load data
#' data = LPP2005REC
#' print (mi_cont (data, 3, 'ksg1'))
#' print (mi_cont (data, 3, 'ksg2'))

 mi_cont <- function (df, k = 3, algo = 'ksg1')
 {
    #Info =  Module ('InfoEntropy', PACKAGE = "NlinTS")
    M = t(df)
    v = .Info $ mutualInformation_cont (M, k, algo);
    return (v);
 }

#' Continuous  Transfer Entropy
#'
#' @details Computes the continuous Transfer Entropy from the second time series to the one using the Kraskov approximation.
#' @param X Integer vector, first time series.
#' @param Y Integer vector, the second time series.
#' @param p Integer, the lag parameter to use for the first vector, (p = 1 by default).
#' @param q Integer the lag parameter to use for the first vector, (q = 1 by default).
#' @param k  Integer argument, the number of neighbours.
#' @param normalize Logical argument  for the option of normalizing value of TE (transfer entropy) (FALSE by default).
#' This normalization is different from the discrete case, because, here the term H (X(t)| X(t-1), ..., X(t-p)) may be negative.
#' Consequently, to get the normalized TE (NTE), we divide TE by H0 - H (X(t)| X(t-1), ..., X(t-p)), where H0 is the entropy of X.
#' @references{
#'   \insertRef{kraskov2004estimating}{NlinTS}
#' }
#' @importFrom Rdpack reprompt
#' @examples
#' library (timeSeries)
#' library (NlinTS)
#' #load data
#' data = LPP2005REC
#' te = te_cont (data[,1], data[,2], 1, 1, 3)
#' print (te)
 te_cont <- function (X, Y, p = 1, q = 1, k = 3, normalize = FALSE)
 {
     #Info =  Module ('InfoEntropy', PACKAGE = "NlinTS")
     v = .Info $ transferEntropy_cont (X, Y, p, q, k, normalize);
     return (v);
 }





