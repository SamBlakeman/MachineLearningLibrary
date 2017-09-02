//
//  MachineLearningModel.cpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 02/09/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#include "MachineLearningModel.hpp"
#include <iostream>
#include <cmath>
#include <numeric>


double MachineLearningModel::CalculateRSquared(const vector<vector<double>>& X, const vector<double>& Y)
{
    if(CostFun == CrossEntropy)
    {
        cout << "A classification problem has no R squared" << endl;
        return 0;
    }
    
    double RSquared = 0;
    
    // Residuals sum of squares
    double RSS = 0;
    vector<double> Predictions = Predict(X);
    
    for(int i = 0; i < Predictions.size(); ++i)
    {
        RSS += pow(Y[i] - Predictions[i], 2);
    }
    
    
    // Total sum of squares
    double TSS = 0;
    double sum = accumulate(Y.begin(), Y.end(), 0);
    double mean = sum/Y.size();
    
    for(int i = 0; i < Y.size(); ++i)
    {
        TSS += pow(Y[i] - mean, 2);
    }
    
    // Calculate R squared
    RSquared = 1 - (RSS/TSS);
    
    return RSquared;
}


double MachineLearningModel::CalculateAccuracy(const vector<vector<double>>& X, const vector<double>& Y)
{
    if(CostFun == SumOfSquaredErrors)
    {
        cout << "A regression problem has no accuracy" << endl;
        return 0;
    }
    
    double Accuracy;
    vector<double> Predictions = Predict(X);
    
    int numCorrect = 0;
    
    for(int i = 0; i < X.size(); ++i)
    {
        if(Y[i] == Predictions[i])
        {
            ++numCorrect;
        }
    }
    
    Accuracy = (double(numCorrect)/X.size())*100;
    
    return Accuracy;
}