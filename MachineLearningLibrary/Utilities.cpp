//
//  Utilities.cpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 02/05/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#include "Utilities.hpp"
#include <iostream>
#include <numeric>
#include <cmath>
#include <string>
#include <sstream>
#include <fstream>

void Utilities::PrintVector(const vector<double>& Vec)
{
    for(int i = 0; i < Vec.size(); ++i)
    {
        cout << Vec[i] << endl;
    }
    cout << endl;
    
    return;
}

void Utilities::Print2DVector(const vector<vector<double>>& Vec)
{
    for(int i = 0; i < Vec.size(); ++i)
    {
        for(int j = 0; j < Vec[i].size(); ++j)
        {
            cout << Vec[i][j] << "\t";
        }
        cout << endl;
    }
    return;
}




// Matrix Operations

vector<vector<double>> Utilities::Transpose(const vector<vector<double>>& Vec1)
{
    vector<vector<double>> Result;
    
    int numRows = (int)Vec1[0].size();
    int numCols = (int)Vec1.size();
    
    vector<double> r(numCols);
    
    for(int row = 0; row < numRows; ++row)
    {
        for(int col = 0; col < numCols; ++ col)
        {
            r[col] = Vec1[col][row];
        }
        
        Result.push_back(r);
    }
    
    return Result;
}




// Products

vector<vector<double>> Utilities::Product(const vector<vector<double>>& Vec1, const vector<vector<double>>& Vec2)
{
    if(Vec1[0].size() != Vec2.size())
    {
        cout << "Error - Matrix Dimension Mismatch";
        return Vec1;
    }
    
    int numRows = (int)Vec1.size();
    int numCols = (int)Vec2[0].size();
    
    vector<vector<double>> Result;
    
    for(int row = 0; row < numRows; ++row)
    {
        vector<double> ResultCol(numCols);
        
        for(int col = 0; col < numCols; ++col)
        {
            vector<double> a = Vec1[row];
            vector<double> b(Vec2.size());
            
            for(int i = 0; i < Vec2.size(); ++i)
            {
                b[i] = Vec2[i][col];
            }
            
            double start = 0;
            ResultCol[col] = inner_product(begin(a), end(a), begin(b), start);
        }
        Result.push_back(ResultCol);
    }
    return Result;
}

// Treats second vector as a column vector
vector<double> Utilities::Product(const vector<vector<double>>& Vec1, const vector<double>& Vec2)
{
    if(Vec1[0].size() != Vec2.size())
    {
        cout << "Error - Matrix Dimension Mismatch";
        return Vec2;
    }
    
    int numRows = (int)Vec1.size();
    
    vector<double> Result(numRows);
    
    for(int row = 0; row < numRows; ++row)
    {
            vector<double> a = Vec1[row];

            Result[row] = inner_product(begin(a), end(a), begin(Vec2), 0.0);
    }
    return Result;
}

// treats first vector as a row vector
vector<double> Utilities::Product(const vector<double>& Vec1, const vector<vector<double>>& Vec2)
{
    if(Vec1.size() != Vec2.size())
    {
        cout << "Error - Matrix Dimension Mismatch";
        return Vec1;
    }
    
    int numCols = (int)Vec2[0].size();
    
    vector<double> Result(numCols);
    
    for(int col = 0; col < numCols; ++col)
    {
        vector<double> b(Vec2.size());
        
        for(int i = 0; i < Vec2.size(); ++i)
        {
            b[i] = Vec2[i][col];
        }
        
        Result[col] = inner_product(begin(Vec1), end(Vec1), begin(b), 0.0);
    }
    return Result;
}

double Utilities::Product(const vector<double>& Vec1, const vector<double>& Vec2)
{
    if(Vec1.size() != Vec2.size())
    {
        cout << "Error - Matrix Dimension Mismatch";
        return 0.f;
    }
    
    double Result;

    Result = inner_product(begin(Vec1), end(Vec1), begin(Vec2), 0.0);
    
    return Result;
}




// Vector Arithmetic
vector<double> Utilities::VecAdd(const vector<double>& Vec1, const vector<double>& Vec2)
{
    vector<double> Result(Vec1.size());
    
    for(int i = 0; i < Vec1.size(); ++i)
    {
        Result[i] = Vec1[i] + Vec2[i];
    }
    
    return Result;
}

vector<double> Utilities::VecSub(const vector<double>& Vec1, const vector<double>& Vec2)
{
    vector<double> Result(Vec1.size());
    
    for(int i = 0; i < Vec1.size(); ++i)
    {
        Result[i] = Vec1[i] - Vec2[i];
    }
    
    return Result;
}


vector<double> Utilities::VecMult(const vector<double>& Vec1, const vector<double>& Vec2)
{
    vector<double> Result = Vec1;
    
    for(int i = 0; i < Result.size(); ++i)
    {
        Result[i] *= Vec2[i];
    }
    
    return Result;
}




vector<vector<double>> Utilities::MatAdd(const vector<vector<double>>& Mat1, const vector<vector<double>>& Mat2)
{
    vector<vector<double>> Result = Mat1;
    
    for(int r = 0; r < Result.size(); ++r)
    {
        for(int c = 0; c < Result[0].size(); ++c)
        {
            Result[r][c] += Mat2[r][c];
        }
    }
    
    return Result;
}


vector<vector<double>> Utilities::MatSub(const vector<vector<double>>& Mat1, const vector<vector<double>>& Mat2)
{
    vector<vector<double>> Result = Mat1;
    
    for(int r = 0; r < Result.size(); ++r)
    {
        for(int c = 0; c < Result[0].size(); ++c)
        {
            Result[r][c] -= Mat2[r][c];
        }
    }
    
    return Result;
}


vector<vector<double>> Utilities::MatMult(const vector<vector<double>>& Mat1, const vector<vector<double>>& Mat2)
{
    vector<vector<double>> Result = Mat1;
    
    for(int r = 0; r < Result.size(); ++r)
    {
        for(int c = 0; c < Result[0].size(); ++c)
        {
            Result[r][c] *= Mat2[r][c];
        }
    }
    
    return Result;
}



double Utilities::GetVecMean (const vector<double>& Vec)
{
    return accumulate(Vec.begin(), Vec.end(), 0.0) / Vec.size();
}

double Utilities::GetVecStd (const vector<double>& Vec, double Mean)
{
    double variance = 0;
    
    for(int i =0; i < Vec.size(); ++i)
    {
        variance += pow((Vec[i] - Mean), 2);
    }
    
    variance = variance / (Vec.size()-1);
    
    return sqrt(variance);
}


// Scalar Transformations

vector<double> Utilities::ScalarMult(const vector<double>& Vec, double scalar)
{
    vector<double> Result = Vec;
    
    for(int i = 0; i < Result.size(); ++i)
    {
        Result[i] *= scalar;
    }
    
    return Result;
}


vector<double> Utilities::ScalarDiv(const vector<double>& Vec, double scalar)
{
    vector<double> Result = Vec;
    
    for(int i = 0; i < Result.size(); ++i)
    {
        Result[i] /= scalar;
    }
    
    return Result;
}

vector<double> Utilities::ScalarAdd(const vector<double>& Vec, double scalar)
{
    vector<double> Result = Vec;
    
    for(int i = 0; i < Result.size(); ++i)
    {
        Result[i] += scalar;
    }
    
    return Result;
}

vector<double> Utilities::ScalarSub(const vector<double>& Vec, double scalar)
{
    vector<double> Result = Vec;
    
    for(int i = 0; i < Result.size(); ++i)
    {
        Result[i] -= scalar;
    }
    
    return Result;
}


vector<double> Utilities::ScalarSub(double scalar, const vector<double>& Vec)
{
    vector<double> Result(Vec.size(),scalar);
    
    for(int i = 0; i < Result.size(); ++i)
    {
        Result[i] -= Vec[i];
    }
    
    return Result;
    
}


vector<vector<double>> Utilities::ScalarSub(double scalar, const vector<vector<double>>& Mat)
{
    vector<vector<double>> Result(Mat.size(),vector<double>(Mat[0].size(),scalar));
    
    for(int r = 0; r < Result.size(); ++r)
    {
        for(int c = 0; c < Result[0].size(); ++c)
        {
            Result[r][c] -= Mat[r][c];
        }
    }
    
    return Result;
}

vector<vector<double>> Utilities::ScalarMult(double scalar, const vector<vector<double>>& Mat)
{
    vector<vector<double>> Result(Mat.size(),vector<double>(Mat[0].size(),scalar));
    
    for(int r = 0; r < Result.size(); ++r)
    {
        for(int c = 0; c < Result[0].size(); ++c)
        {
            Result[r][c] *= Mat[r][c];
        }
    }
    
    return Result;
}


// Reading txt files
vector<vector<double>> Utilities::ReadCSVFeatureVector(const char* FileName)
{
    vector<vector<double>> Result;
    
    ifstream file;
    file.open(FileName);
    
    // Check that it opens
    if(!file.is_open())
    {
        cout << "file not open" << endl;
        return Result;
    }
    
    // Get contents
    vector<double> row;
    
    string line;
    
    while (file.good())
    {
        while(getline(file,line))
        {
            stringstream linestream(line);
            string str;
            double num;
            
            while(getline(linestream,str,','))
            {
                stringstream numStream(str);
                numStream >> num;
                row.push_back(num);
            }
            Result.push_back(row);
            row.clear();
        }
    }
    
    cout << ".txt file read successfully" << endl;
    
    return Result;
}

void Utilities::SaveVectorAsCSV(const vector<double>& Vec, const char* FileName)
{
    ofstream myFile(FileName);
    
    for(int i = 0; i < Vec.size(); ++i)
    {
        myFile << Vec[i] << ',';
    }
    
    if(myFile.is_open())
    {
        myFile.close();
    }
    
    cout << ".txt file saved successfully" << endl;
    
    return;
}





