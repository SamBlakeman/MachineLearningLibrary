//
//  Utilities.hpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 02/05/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#ifndef Utilities_hpp
#define Utilities_hpp

#include <stdio.h>
#include <vector>

#endif /* Utilities_hpp */

using namespace std;

class Utilities
{
public:
    static void PrintVector(const vector<double>& Vec);
    static void Print2DVector(const vector<vector<double>>& Vec);
    
    // Matrix operations
    static vector<vector<double>> Transpose(const vector<vector<double>>& Vec1);
    
    // Products
    static vector<vector<double>> Product(const vector<vector<double>>& Vec1, const vector<vector<double>>& Vec2);
    static vector<double> Product(const vector<vector<double>>& Vec1, const vector<double>& Vec2);
    static vector<double> Product(const vector<double>& Vec1, const vector<vector<double>>& Vec2);
    static double Product(const vector<double>& Vec1, const vector<double>& Vec2);
    
    // Vector Arithmetic
    static vector<double> VecAdd(const vector<double>& Vec1, const vector<double>& Vec2);
    static vector<double> VecSub(const vector<double>& Vec1, const vector<double>& Vec2);
    
    // Vector Statistics
    static double GetVecMean (const vector<double>& Vec);
    static double GetVecStd (const vector<double>& Vec, double Mean);
    
    // Scalar Transformations
    static vector<double> ScalarMult(const vector<double>& Vec, double scalar);
    static vector<double> ScalarMult(const vector<double>& Vec1, const vector<double>& Vec2);
    static vector<double> ScalarDiv(const vector<double>& Vec, double scalar);
    static vector<double> ScalarAdd(const vector<double>& Vec, double scalar);
    static vector<double> ScalarAdd(const vector<double>& Vec1, const vector<double>& Vec2);
    static vector<double> ScalarSub(const vector<double>& Vec, double scalar);
    static vector<double> ScalarSub(double scalar, const vector<double>& Vec);
    static vector<double> ScalarSub(const vector<double>& Vec1, const vector<double>& Vec2);
    
    // Reading and saving txt files (comma separated and returns between examples)
    static vector<vector<double>> ReadCSVFeatureVector(const char* FileName);
    static void SaveVectorAsCSV(const vector<double>& Vec, const char* FileName);
    
};
