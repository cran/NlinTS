% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/modues.R
\name{causality.test}
\alias{causality.test}
\title{The Granger causality test}
\usage{
causality.test(ts1, ts2, lag, diff = FALSE)
}
\arguments{
\item{ts1}{Numerical dataframe containing one variable}

\item{ts2}{Numerical dataframe containing one variable}

\item{lag}{The lag parameter}

\item{diff}{Logical argument for the option of making data stationary}
}
\value{
summary ():  shows the test results

F-test (): returns the value of the test
}
\description{
The Granger causality test
}
\details{
The test evaluates if the second time series causes the first one using the Granger test of causality.
}
\examples{
library (timeSeries) # to extract time series
library (NlinTS)
data = LPP2005REC
model = causality.test (data[,1], data[,2], 2)
model$summary ()
}
\references{
{
  \insertRef{granger1980}{NlinTS}
}
}