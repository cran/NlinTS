\name{NlinTS-package}
\alias{NlinTS-package}
\docType{package}
\title{Non linear time series forecasting and causality detection}
\description{
This package is for non-linear time series forecasting and causality detection. It contains models based on an non-linear version of the VAR model, based on a multi-layer perceptron neural network.
}
\details{
 The main implementations of this package are a neural network VAR model and a non-linear Granget causality test.
 They can be considered as an alternative to the VAR model and the linear Granger causality test respectively.
 It contains also an implementation of the classical Granger causality test, with an additional functionality that consists in allowing the possibility of transforming data to stationary (in case of one of them or both are non stationary) before applying the VAR models used in the test. The Augmented Dickey-Fuller test is also provided.
}
\author{
Youssef Hmamouche, hmamoucheyussef@gmail.com.

Maintainer: Youssef Hmamouche <hmamoucheyussef@gmail.com>
}
%\references{
 % This optional section can contain literature or other references for
  %background information.
%}
\keyword{ package, time series forecasting, causality detection, neural networks }
%\seealso{
 % Optional links to other man pages
%}
