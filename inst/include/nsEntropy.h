#ifndef TRANSFERENTROPY_H
#define TRANSFERENTROPY_H

#include <vector>

#define EXP exp(1)

using namespace std;

double myLOG (double x, std::string log = "log2");
double digamma (double x);

typedef std::vector<int> VectInt;
typedef std::vector<double> VectDouble;
typedef std::vector<std::vector<int>> MatInt;
typedef std::vector<std::vector<double>> MatDouble;

namespace nsEntropy{
VectInt count (const std::vector<int> & X);
MatInt count (const MatInt & X);

VectDouble minMax (const VectDouble & vect);
MatDouble minMax (const MatDouble & mat);
void normalize (MatDouble & mat);

double entropy (const std::vector<int> & Vect, std::string log = "log");

double joinProba (std::vector<int> X, std::vector<int> Y, int x, int y);
double joinProba (MatInt Y, VectInt y);

double Proba (std::vector<int> X,  int x);
MatInt generateTuples (const MatInt & M);


double joinEntropy (const std::vector<int> & X, const std::vector<int> & Y, std::string log = "log");
double joinEntropy (const MatInt & Mat, std::string base = "log");


double condEntropy (const VectInt & X, const VectInt & Y, std::string log = "log");
double condEntropy (const VectInt & X, const MatInt & Y, std::string base = "log");

double mutualInformation (const MatInt & X, std::string log = "log");
double mutualInformation (const std::vector<int> & X, const std::vector<int> & Y, std::string log = "log");

double transferEntropy (const VectInt & X, const VectInt & Y, int p, int q, std::string base = "log", bool normalize = false);



/*--------------- contnious variables ----------------*/
double dist (double x, double y);
double dist (VectDouble X, VectDouble Y);

double entropy (const VectDouble & V, int k);
double joinEntropy (const MatDouble & M, int k);

double mutualInformation (const MatDouble & M, int k, std::string alg = "ksg1");
double transferEntropy (const VectDouble & X, const VectDouble & Y, int p=1, int q=1, int k=3, bool normalize = true);
double transferEntropy_ksg (const VectDouble & X, const VectDouble & Y, int p=1, int q=1, int k=3);

/*********************************************************/
// compute the number of neighbors of a given axis or set of axis
template <class type>
VectInt computeNbOfNeighbors (const vector<type> & X, VectDouble radius, bool equal = false)
{
	unsigned N = X. size ();
	VectInt NAx (N, 0);

	// we count the number of points Xj whose distance from Xi is strictly less than radius[i],
	for (unsigned i = 0; i < N; i ++)
		for (unsigned j = 0; j < N; j ++)
		{
			if (j != i)
			{
				if (dist (X[i], X[j]) < radius [i] and equal == 0)
					NAx[i] += 1;
				else if (dist (X[i], X[j]) <= radius [i] and equal)
					NAx[i] += 1;
			}
		}
	return NAx;
}

/*********************************************************/
VectInt nbOfNeighborsInRectangle (const MatDouble & X, const MatDouble X1, const MatDouble X2, 
						           const VectDouble & distances);


/*********************************************************/
template <class type>
MatDouble distanceMatrix (const vector <type> & V)
{
	unsigned n = V.size ();
	MatDouble M (n);

	for (unsigned i = 0; i < n; i++)
		M[i]. resize (n, 0);
	
	for (unsigned i = 0; i < n - 1; i++)
		for (unsigned j = i + 1; j < n; j++){
			M[i][j] = dist (V[i], V[j]);
			M[j][i] = M[i][j];
		}

	return M;
};

 /*********************************************************/
template <class type>
vector<double> kNearest (const vector<type> & V, int k)
{
	MatDouble distMat = distanceMatrix (V);
	vector<double> result (V.size ());

	for (unsigned i = 0; i < V. size (); i++)
	{
			std::sort (distMat[i].begin(), distMat[i].end());
			result [i] = distMat[i][k];
	}

	return result;
}

/***************************************************************/
template <class type>
vector<vector<type>> lagg (const vector<type> & V, unsigned  p, bool c = 0)
{
    unsigned int  N = V.size ();
    vector<type> P ;

    // current = 1 if we want to add the  current values of the variable with the lagged ones
    vector<vector<type>> M (N - p);
    for (unsigned j = 0 ; j < N - p ; j++)
    	M[j]. resize (p + c, 0);
   
    for (unsigned i = 0 ; i < N - p ; i++)
    	for (unsigned j = 0; j < p + c ; ++j)   
    		M[i][j] = V[i + (p - 1 + c) - j];	    
    
    return M;
}

/*********************************************************/
template<class type>
vector<vector<type>> getCols (const vector<vector<type>> & M, const vector<type> & cols)
{
	vector<vector<type>> SmallM (M. size ()) ;

	for (unsigned i = 0; i < M. size (); ++i)
		for (double idx : cols)
				SmallM[i]. push_back (M[i][idx]);

	return SmallM;
};

/*********************************************************/
template <class type>
vector<type> getCol (const vector<vector<type>> & M, int col)
{
	vector<type> Vec (M. size ()) ;

	for (unsigned i = 0; i < M. size (); ++i)
				Vec[i] =  M[i][col];

	return Vec;
};

/*********************************************************/
template <class type>
void show (const std::vector<type> & Vect){
	for (auto & val:Vect)
		std::cout << val << " ";
	std::cout << std::endl;}

/*********************************************************/
template <class type>
void showM (const std::vector<std::vector<type>> & Vect){
	for (auto & val:Vect)
		show (val);
	std::cout << std::endl;} 
}


#endif
