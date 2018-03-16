/**
 * @authors Hmamouche Youssef
 **/

#include "../inst/include/CVARMLPExport.h"

using namespace std;

CRcppExport::CRcppExport (Rcpp::DataFrame Df, unsigned p, Rcpp::IntegerVector size_Layers, unsigned iters, bool bias)
{
    
    vector<int> sizeOfLayers_ = Rcpp::as <vector<int> > (size_Layers);
    
    vector<vector<double> > Mat = Rcpp::as < vector<vector<double> > > (Df);
    unsigned nc = Mat.size ();
    Struct::CMatDouble M (nc);
    
    unsigned i = 0;
    
    // Conversion from R dataframe to  C++ table.
    while(i < nc)
    {
        for (auto Value = Mat[i].begin () ; Value != Mat[i].end () ; ++Value)
            M[i].push_back (*Value);
        i++;
    }

    // Create the model
    Obj = CNeuralNVarModel (sizeOfLayers_, p, bias);
    Obj.fit (M, iters);
    //fit (Df, iters);
    }
/****************************************************/
void CRcppExport::fit (Rcpp::DataFrame Df, unsigned iters)
{
    vector<vector<double> > Mat = Rcpp::as < vector<vector<double> > > (Df);
    unsigned nc = Mat.size ();
    Struct::CMatDouble M (nc);
   
    unsigned i = 0;
    
    // Conversion from R dataframe to  C++ table.
    while(i < nc)
    {
        for (auto Value = Mat[i].begin () ; Value != Mat[i].end () ; ++Value)
        M[i].push_back (*Value);
        i++;
    }
    
    Obj.fit (M, iters);
}
/****************************************************/

Rcpp::DataFrame CRcppExport::forecast (Rcpp::DataFrame DF)
  {
      
    unsigned i = 0;
    vector<vector<double> > Mat = Rcpp::as < vector<vector<double> > > (DF);
    unsigned  nc = Mat.size();
    Struct::CMatDouble P (nc);
     
     // Conversion des données dataFrame en matrice en C++
     while(i < nc)
     {
         for (auto Value = Mat[i].begin () ; Value != Mat[i].end () ; ++Value)
         {
             P[i].push_back (*Value);
         }
         i++;
     }
 
      Rcpp::List list (nc);
      Rcpp::DataFrame dataFrame;
      
      Struct::CMatDouble SM = Obj.forecast (P);
      
      for ( unsigned j = 0; j < nc; j++)
          list[j] = Rcpp::wrap( SM[j].begin(), SM[j].end() ) ;
      
      dataFrame = list;
      
      Rcpp::CharacterVector ch = DF.names ();
      
      if (ch.size() > 0)
          dataFrame.names() = ch;
      
      return dataFrame;

 }
/****************************************************/
void CRcppExport::train (Rcpp::DataFrame DF)

{
    vector<vector<double> > Mat = Rcpp::as < vector<vector<double> > > (DF);
    int i = 0, nc = Mat.size();
    Struct::CMatDouble P (nc);
    
    // Conversion des données dataFrame en matrice en C++
    while(i < nc)
    {
        for (auto Value = Mat[i].begin () ; Value != Mat[i].end () ; ++Value)
            P[i].push_back (*Value);
        i++;
    }
    
    Obj.train (P);
}
/************ model accuracy ********************/
Rcpp::NumericVector CRcppExport::getSSR ()
{
    return (Rcpp::wrap (Obj.getSSR()));
}



