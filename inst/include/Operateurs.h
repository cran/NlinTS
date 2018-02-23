/**
 * @authors Hmamouche Youssef
 **/

#ifndef OPERATEURS_H
#define OPERATEURS_H

#include <Rcpp.h>

#include "Struct.h"
#include "Exception.h"

using namespace Struct;

/******** Produit vecteur--vecteur *********/
void MultCVDouble (const CVDouble & A, const CVDouble & B, CVDouble & Res) throw (Exception);

/********* Produit Matric--vecteur *********/
void MultCVDouble (const CMatDouble & A, const CVDouble & B, CVDouble & Res) throw (Exception)
;

/*************** Produit Matrice--Matrice  ***************/
void MultCVDouble (const CMatDouble & A, const CMatDouble & B, CMatDouble & Res) throw (Exception);

#endif // OPERATEURS_H
