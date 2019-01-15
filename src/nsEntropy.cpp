#include <iostream>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include "../inst/include/nsEntropy.h"

using namespace std;

/*****************************************************/
// Extract distinct values

namespace nsEntropy {

double myLOG (double x, std::string base)
{
  if (base == "loge")
    return (log2 (x) / log2(EXP));
  else if (base == "log10")
    return (log2 (x) / log2(10));
  else if (base == "log2")
    return log2 (x);
  else
    return log2 (x);
}


double digamma (double x)
{
  double a = 0, b, c;


  while (x <= 5) 
  { 
    a -= 1 / x;
    x += 1;
  }

  b = 1 / (x * x);

  c = b * (-1/12.0 + 
      b * (1/120.0 + 
      b * (-1/252.0 + 
      b * (1/240.0 + 
      b * (-1/132.0 +
      b * (691/32760.0 + 
      b * (-1/12.0 + 
      b * 3617/8160.0)))))));

  return (a + log (x) - 0.5 / x + c);
}

	// Min and max values of columns of a matrix
VectDouble minMax (const VectDouble & vect)
{
	VectDouble result (2);
	result [0] = vect [0];
	result [1] = vect [0];

	for (unsigned i = 1; i < vect. size (); ++i)
	{
		if (vect [i] < result [0])
			result [0] = vect [i];

		if (vect [i] > result [1])
			result [1] = vect [i];
	}
	return result;
}
MatDouble minMax (const MatDouble & mat)
{
	MatDouble result (mat[0]. size ());
	for (unsigned j = 0; j < mat[0]. size (); ++j)
	{
		result [j] = minMax (getCol (mat, j));
	}
	return result;
}

void normalize (MatDouble & mat)
{
	MatDouble min_max = minMax (mat);
	unsigned m = mat[0]. size (), n = mat. size ();

		for (unsigned j = 0; j < m; ++j)
		{
			if (min_max[j][0] != min_max [j][1])
			{
				for (unsigned i = 0; i < n; ++i)
					mat[i][j] = (mat[i][j] - min_max[j][0]) / (min_max[j][1] - min_max[j][0]);
			}

		}
}

vector<int> count (const std::vector<int> & X)
{
	std::vector<int> Vect (X);
	std::vector<int>::iterator it;
	std::sort (Vect.begin(), Vect.end());
	it = std::unique (Vect. begin (), Vect. end ()); 

		// Distinct values
	Vect.resize (std::distance (Vect.begin(), it));
	return Vect;
}

/*****************************************************/
MatInt count (const MatInt & X)
{
	vector<vector<int>> Vect (X);
	vector<vector<int>>::iterator it;
	std::sort (Vect.begin(), Vect.end());
	it = std::unique (Vect. begin (), Vect. end ()); 

		// Distinct values
	Vect.resize (std::distance (Vect.begin(), it));
	return Vect;
}

/*****************************************************/
double joinProba (vector<int> X, vector<int> Y, int x, int y)
{
	double J = 0;
	for (unsigned i = 0; i < X.size (); ++i)
	{
		if (X[i] == x and Y[i] == y)
			J++;
	}
	return (J / X.size ());
}

/*****************************************************/
double joinProba (MatInt Y, VectInt y)
{
	double J = 0;
	unsigned j;

	for (unsigned i = 0; i < Y.size (); ++i)
	{
		for (j = 0; j < Y [0] .size (); ++j)
			if (Y[i][j] != y[j]) break;	

		if (j == Y [0].size ())
			J++;
	}
	return (J / Y.size ());
}

/*****************************************************/
double Proba (vector<int> X, int x)
{
	double P = 0;
	for (unsigned i = 0; i < X.size (); ++i)
	{
		if (X[i] == x)
			P++;
	}
	return (P / X.size ());
}


/*****************************************************/
double entropy (const VectInt & X, string base)
{
	double E = 0, x;

	vector<int> Vect = count (X);
	unsigned n = Vect.size ();
	for (unsigned i = 0; i < n; i++)
	{
			x = Proba (X, Vect[i]);
			if (x > 0)
				E += x  * myLOG (x, base) ;
	}
	
	return -E;
}
/*****************************************************/
double joinEntropy (const VectInt & X1, const VectInt & X2, string base)
{
	double J = 0;
	double x;

	vector<int> X = count (X1);
	vector<int> Y = count (X2);
	unsigned n = X. size ();
	unsigned m = Y. size ();

	for (unsigned i = 0; i < n; i++)
	{
		for (unsigned j = 0; j < m; j++) {
			x = joinProba (X1, X2, X[i], Y[j]);

			if (x > 0)
				J += x  * myLOG (x, base);
		}
	}
	return -J;
}

/*****************************************************/
double joinEntropy (const MatInt & Mat, string base)
{
	double J = 0;
	double x;

	MatInt tuples = count (Mat);

	for (auto tuple : tuples)
	{			
		x = joinProba (Mat, tuple);
		if (x > 0)
			J += x  * myLOG (x, base);
	}
	return -J;
}

/*****************************************************/
double condEntropy (const VectInt & X, const VectInt & Y, string base)
{
	return joinEntropy (X, Y, base) - entropy (Y, base);
}

/*****************************************************/
double condEntropy (const VectInt & X, const MatInt & Y, string base)
{
	MatInt M = Y;
	M .push_back (X);

	return (joinEntropy (M, base) - joinEntropy (Y, base));
}

/*****************************************************/
double mutualInformation (const VectInt & X, const VectInt & Y, std::string base)
{
	return  (entropy (X, base) + entropy (Y, base)  - joinEntropy (X, Y, base));
}

double mutualInformation (const MatInt & X, std::string base)
{
	double mi = 0;
	VectInt vect;

	for (unsigned i = 0; i < X [0]. size (); ++i)
	{
		vect = getCol (X, i);
		mi += entropy (vect, base);
	}

	mi -= joinEntropy (X, base);
	return  mi;
}

double transferEntropy (const VectInt & X, const VectInt & Y, int p, int q, std::string base, bool normalize)
{
	double te, denom;
	MatInt Xp, Xm, Ym, XmYm, XpYm;
	Xm = lagg (X, p, 0);
	Xp = lagg (X, p, 1);
	Ym = lagg (Y, q, 0);

	XmYm = Xm;
	XpYm = Xp;

	// Resize the join matrix to have the same lenght if p # q
	if ((p - q) < 0)
	{
		XmYm. erase (XmYm.begin (), XmYm.begin()  + q - p);
		XpYm. erase (XpYm.begin (), XpYm.begin()  + q - p);
	}
	if ((p - q) > 0)
		Ym. erase (Ym.begin (), Ym.begin()  + p - q);

	unsigned N = Ym. size ();

	for (unsigned i = 0 ; i < N; ++i)
		for (unsigned j = 0 ; j < Ym [0]. size (); ++j){
			XmYm [i]. push_back ( Ym [i][j] );
			XpYm [i]. push_back ( Ym [i][j] );
		}

	denom = joinEntropy (Xp, base) - joinEntropy (Xm, base);
	te = denom - joinEntropy (XpYm, base) + joinEntropy (XmYm, base);

	if (normalize and denom != 0)
			te = te / denom;

	return te;
}

/*********************************************************/
/*--------------- continuous variables ------------------*/
/*********************************************************/

double dist (double x, double y) {
	return abs (x - y);
};

double dist (VectDouble X, VectDouble Y)
{
	double distance = 0;

	for (unsigned i = 0; i < X. size (); ++i){
		  if (distance < abs (X[i] - Y[i]))
			distance = abs (X[i] - Y[i]);
		}
	return  distance;
};


 /*********************************************************/

double entropy (const VectDouble & V, int k)
{
	double E = 0;
	unsigned N = V. size ();
	double sum = 0;
	double cd = 2;

	VectDouble distances = kNearest (V, k);

	for (unsigned i = 0; i < N; i ++)
		sum += myLOG (2 * distances [i], "log2");

	sum = sum / N; 
	E = digamma (N) - digamma (k) +  sum  + myLOG (cd, "log2"); // + log2 (cd) ;// + sum;
	return E ;
}

 /*********************************************************/
double joinEntropy (const MatDouble & M, int k)
{
	double E = 0;
	unsigned N = M. size ();
	double sum = 0;

	unsigned d = M[0]. size ();

	VectDouble distances = kNearest (M, k);

	for (unsigned i = 0; i < N; i ++)
		sum += myLOG (2 * distances [i], "loge");

	sum = sum * d / N; 

	E = digamma (N) - digamma (k) +  sum ;
	return E ;
}



/*****************************************************/
double mutualInformation (const MatDouble & M, int k, string alg)
{
	double mi = 0;
	unsigned N = M. size ();
	double sum = 0;

	VectInt NX, NY;

	VectDouble radius = kNearest (M, k);
	//show (radius);
	//NX = computeNbOfNeighbors (getCol (M, 0), distances);
	//NY = computeNbOfNeighbors (getCol (M, 1), distances);

	if (alg == "ksg1")
	{
		NX = computeNbOfNeighbors (getCol (M, 0), radius, true);
		NY = computeNbOfNeighbors (getCol (M, 1), radius, true);

		for (unsigned i = 0; i < N; i ++)
			sum += digamma (NX[i] + 1) + digamma (NY[i] + 1);

		sum = sum  / N; 
		mi = digamma (k) + digamma (N) - sum;
	}

	else if (alg == "ksg2")
	{
		NX = computeNbOfNeighbors (getCol (M, 0), radius, true);
		NY = computeNbOfNeighbors (getCol (M, 1), radius, true);

		for (unsigned i = 0; i < N; i ++)
			sum += digamma (NX[i]) + digamma (NY[i]);

		sum = sum  / N; 
		mi = digamma (k) - (1.0 / k) + digamma (N) - sum;
	}


	return mi;
}

/***************************************************************/
// Tranfer entropy from Y to X
double transferEntropy (const VectDouble & X, const VectDouble & Y, int p, int q, int k, bool normalize)
{
	double te, sum = 0;


	MatDouble Xm, Xp, Ym, XpYm, XmYm;
	VectInt NXm, NXmYm, NXpYm, NXp; 

	Xm = lagg (X, p, 0);
	Ym = lagg (Y, q);
	Xp = lagg (X, p, 1);

	// Resize the join matrix to have the same lenght if p # q
	if ((p - q) < 0)
	{
		Xm. erase (Xm.begin (), Xm.begin()  + q - p);
		Xp. erase (Xp.begin (), Xp.begin()  + q - p);
	}
	if ((p - q) > 0)
		Ym. erase (Ym.begin (), Ym.begin()  + p - q);

	//the number of observation of lagged variables : n - max (p, q)
	unsigned N = Ym.size ();
	XpYm = Xp;
	XmYm = Xm;

	for (unsigned i = 0 ; i < N; ++i)
		for (unsigned j = 0 ; j < Ym [0]. size (); ++j){
			XmYm [i]. push_back ( Ym [i][j] );
			XpYm [i]. push_back ( Ym [i][j] );
		}

	//  distances from k neighbors of the join matrix Xp  (Xcurrent + Xpassed + Ypassed)
	VectDouble distances = kNearest (XmYm, k);

	// we count the number of local points relative to  the marginal matrices
	NXmYm = computeNbOfNeighbors (XmYm, distances);
	NXm = computeNbOfNeighbors (Xm, distances);
	NXp = computeNbOfNeighbors (Xp, distances);

	for (unsigned i = 0; i < N; i ++){
		sum += digamma (NXm[i] + 1) - digamma (NXmYm[i] + 1) - digamma (NXp[i] + 1);
	}

	te = digamma (k) + (sum / N) ;

		
	if (normalize == true)
	{
		// Compute H(Xp|Xm), NTE <- TE / (H0 - H(Xp|Xm))
		double denom = 0;
		VectDouble Xt = getCol (Xp, 0);
		VectInt NXt = computeNbOfNeighbors (Xt, distances);

		for (unsigned i = 0; i < N; i ++)
			denom +=  - digamma (NXmYm[i] + 1); //- digamma (NXt[i] + 1);

		denom = (denom / N) + digamma (N) ; //+ digamma (k);;

		te = te / denom;
	}

	return te;
}

VectInt nbOfNeighborsInRectangle (const MatDouble & X, const MatDouble X1, const MatDouble X2, 
						           const VectDouble & distances)
{
	unsigned N = X. size ();
	VectInt Nx (N, 0);

	double dLocalx, dLocaly, dx, dy;

	for (unsigned i = 0 ; i < N; ++i)
	{
		dx = 0;
		dy = 0;
		// Compute dx and dy : edge lengths of the hyper-rectangle
		for (unsigned j = 0 ; j < N; ++j)
		{
			dLocalx = dist (X1[i], X1[j]);
			dLocaly = dist (X2[i], X2[j]);

			if (dist (X[i], X[j]) <= distances [i] and j != i)
			{
				if (dx < dLocalx)
					dx = dLocalx;
				if (dy < dLocaly)
					dy = dLocaly;
			}
		}

		// Count the number of points in the hyper-rectangle
		for (unsigned j = 0 ; j < N; ++j)
		{
			if (dist (X1[i], X1[j]) <= (2 * dx) and  dist (X2[i], X2[j]) <= (2 * dy) )
					Nx[i] += 1;
		}
	}
	return Nx;
}
/**************************************************************************/
double transferEntropy_ksg (const VectDouble & X, const VectDouble & Y, int p, int q, int k)
{
	double te, sum = 0;

	VectInt NXm, NXmYm, NXpYm; 
	MatDouble Xm, Ym, XpYm, XmYm, Xp;	

	Xm = lagg (X, p, 0);
	Ym = lagg (Y, q);
	Xp = lagg (X, p, 1);

	// Resize the join matrix to have the same lenght if p # q
	if ((p - q) < 0)
	{
		Xm. erase (Xm.begin (), Xm.begin()  + q - p);
		Xp. erase (Xp.begin (), Xp.begin()  + q - p);
	}
	if ((p - q) > 0)
		Ym. erase (Ym.begin (), Ym.begin()  + p - q);

	//the number of observation of lagged variables : n - max (p, q)
	unsigned N = Ym.size ();
	XpYm = Xp;
	XmYm = Xm;

	for (unsigned i = 0 ; i < N; ++i)
		for (unsigned j = 0 ; j < Ym [0]. size (); ++j){
			XmYm [i]. push_back ( Ym [i][j] );
			XpYm [i]. push_back ( Ym [i][j] );
		}

	//  distances from k neighbors of the join matrix Xp  (Xcurrent + Xpassed + Ypassed)
	VectDouble distances = kNearest (XpYm, k);

	// We countthe number of points within the hyper-rectangle equal to the Cartesian prod of  XmYm, Xp and Xm
	NXmYm = nbOfNeighborsInRectangle (XpYm, Xm, Ym, distances);
	NXpYm = nbOfNeighborsInRectangle (XpYm, Xp, Xm, distances);
	NXm = nbOfNeighborsInRectangle (XpYm, Xm, Xm, distances);
	//NXm = computeNbOfNeighbors (Xm, distances, true);

	for (unsigned i = 0; i < N; i ++)
		sum += digamma (NXm[i]) - digamma (NXmYm[i]) - digamma (NXpYm[i]) + (1 / NXpYm[i]) + (1 / NXmYm[i]);


	te = digamma (k) - (2 / k) + sum / N; // - (lag * epsilon / N);


	return te;
}

} // end namespace

























