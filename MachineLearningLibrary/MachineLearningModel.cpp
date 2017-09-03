//
//  MachineLearningModel.cpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 02/09/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#include "MachineLearningModel.hpp"
#include "PreProcessing.hpp"
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


KFoldResults MachineLearningModel::KFoldCrossValidation(const vector<vector<double>>& X, const vector<double>& Y, int numFolds)
{
    KFoldResults Results;
    int numExamples = int(X.size());
    
    if(numFolds > numExamples)
    {
        cout << "Error - You have specified more folds than examples";
        return Results;
    }
    
    double TrainRSquared = 0;
    double TrainAccuracy = 0;
    
    double TestRSquared = 0;
    double TestAccuracy = 0;
    
    int quotient = floor(numExamples/numFolds);
    int remainder = numExamples % numFolds;
    
    int q = quotient;
    
    for(int fold = 0; fold < numFolds; ++fold)
    {
        vector<double> YTest;
        vector<vector<double>> XTest;
        
        // Extract Test Matrices
        for(int i = 0; i < quotient; ++i)
        {
            YTest.push_back(Y[q - quotient + i]);
            XTest.push_back(X[q - quotient + i]);
        }
        
        vector<double> YTrain;
        vector<vector<double>> XTrain;
        
        // Extract Training Matrices
        for(int i = 0; i < numExamples; ++i)
        {
            if(i < q - quotient || i > q - 1)
            {
                YTrain.push_back(Y[i]);
                XTrain.push_back(X[i]);
            }
        }
        
        // Normalise
        PreProcessing pp;
        pp.NormaliseFit(XTrain);
        pp.NormaliseTransform(XTrain);
        pp.NormaliseTransform(XTest);
        
        // Peform KFold Cross Validation
        Fit(XTrain,YTrain);
        
        switch(CostFun)
        {
                
            case SumOfSquaredErrors:
                
                TrainRSquared += CalculateRSquared(XTrain, YTrain);
                TestRSquared += CalculateRSquared(XTest, YTest);
                
                break;
                
            case CrossEntropy:
                
                TrainAccuracy += CalculateAccuracy(XTrain, YTrain);
                TestAccuracy += CalculateAccuracy(XTest, YTest);
                
                break;
                
        }
        
        // Update q
        q += quotient;
        
        // In the last fold include the remainder
        if(q == numExamples-remainder)
        {
            q = numExamples;
            quotient += remainder;
        }
    }
            
    Results.TrainingRSquared = TrainRSquared/numFolds;
    Results.TrainingAccuracy = TrainAccuracy/numFolds;
    
    Results.TestRSquared = TestRSquared/numFolds;
    Results.TestAccuracy = TestAccuracy/numFolds;
    
    return Results;
}