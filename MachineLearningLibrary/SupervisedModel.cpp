//
//  SupervisedModel.cpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 02/09/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#include "SupervisedModel.hpp"
#include "PreProcessing.hpp"
#include <iostream>
#include <cmath>
#include <numeric>

double SupervisedModel::CalculatePerformance(const vector<vector<double>>& X, const vector<double>& Y)
{
    double Result = 0;
    
    if(CostFun == SumOfSquaredErrors)
    {
        Result = CalculateRSquared(X, Y);
    }
    else if(CostFun == CrossEntropy)
    {
        Result = CalculateAccuracy(X, Y);
    }
    
    return Result;
}


double SupervisedModel::CalculateRSquared(const vector<vector<double>>& X, const vector<double>& Y)
{
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


double SupervisedModel::CalculateAccuracy(const vector<vector<double>>& X, const vector<double>& Y)
{
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


KFoldResults SupervisedModel::KFoldCrossValidation(const vector<vector<double>>& X, const vector<double>& Y, int numFolds)
{
    KFoldResults Results;
    int numExamples = int(X.size());
    
    if(numFolds > numExamples)
    {
        cout << "Error - You have specified more folds than examples";
        return Results;
    }
    
    double TrainPerformanceSum = 0;
    double TestPerformanceSum = 0;
    vector<double> TrainPerformances(numFolds,0);
    vector<double> TestPerformances(numFolds,0);
    
    int quotient = floor(numExamples/numFolds);
    int remainder = numExamples % numFolds;
    
    int q = quotient;
    double Performance = 0;
    
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
        
        Performance = CalculatePerformance(XTrain, YTrain);
        TrainPerformanceSum += Performance;
        TrainPerformances[fold] = Performance;
        
        Performance = CalculatePerformance(XTest, YTest);
        TestPerformanceSum += Performance;
        TestPerformances[fold] = Performance;
        
        // Update q
        q += quotient;
        
        // In the last fold include the remainder
        if(q == numExamples-remainder)
        {
            q = numExamples;
            quotient += remainder;
        }
    }
            
    Results.TrainMeanPerformance = TrainPerformanceSum/numFolds;
    Results.TrainStdPerformance = 0;
    for(int fold = 0; fold < numFolds; ++fold)
    {
        Results.TrainStdPerformance += pow(TrainPerformances[fold] - Results.TrainMeanPerformance,2);
    }
    Results.TrainStdPerformance /= (numFolds - 1); // Bessel's correction
    
    Results.TestMeanPerformance = TestPerformanceSum/numFolds;
    Results.TestStdPerformance = 0;
    for(int fold = 0; fold < numFolds; ++fold)
    {
        Results.TestStdPerformance += pow(TestPerformances[fold] - Results.TestMeanPerformance,2);
    }
    Results.TestStdPerformance /= (numFolds - 1); // Bessel's correction
    
    return Results;
}

ValidationCurveResults SupervisedModel::ValidationCurve(const vector<vector<double>>& X, const vector<double>& Y, Parameter Param, vector<double> ParamRange, int numFolds)
{
    int numParams = int(ParamRange.size());
    
    vector<double> TrainMeanPerformance(numParams,0);
    vector<double> TrainStdPerformance(numParams,0);
    
    vector<double> TestMeanPerformance(numParams,0);
    vector<double> TestStdPerformance(numParams,0);
    
    ValidationCurveResults Results;
    
    for(int p = 0; p < ParamRange.size(); ++p)
    {
        double value = ParamRange[p];
        
        switch(Param)
        {
            case Lambda:
                SetLambda(value);
                break;
                
            case Alpha:
                SetAlpha(value);
                break;
                
            case Iterations:
                SetIterations(value);
                break;
                
            case Tau:
                SetTau(value);
                break;
                
            case C:
                SetC(value);
                break;
                
            case Var:
                SetVar(value);
                break;
        }
        
        
        // K-fold cross validation
        KFoldResults KResults = KFoldCrossValidation(X, Y, numFolds);
        
        // Record mean and std for parameter value
        TrainMeanPerformance[p] = KResults.TrainMeanPerformance;
        TrainStdPerformance[p] = KResults.TrainStdPerformance;
        
        TestMeanPerformance[p] = KResults.TestMeanPerformance;
        TestStdPerformance[p] = KResults.TestStdPerformance;
        
    }
    
    // Return means and stds
    Results.TrainMeanPerformance = TrainMeanPerformance;
    Results.TrainStdPerformance = TrainStdPerformance;
    
    Results.TestMeanPerformance = TestMeanPerformance;
    Results.TestStdPerformance = TestStdPerformance;
    
    return Results;
}

LearningCurveResults SupervisedModel::LearningCurve(const vector<vector<double>>& X, const vector<double>& Y, int numPoints, int numFolds)
{
    int numFeatures = int(X[0].size());
    int numExamples = int(X.size());
    int numIt = numExamples / numPoints;
    int m = numIt;
    
    vector<double> TrainMeanPerformance(numPoints,0);
    vector<double> TrainStdPerformance(numPoints,0);
    
    vector<double> TestMeanPerformance(numPoints,0);
    vector<double> TestStdPerformance(numPoints,0);
    
    LearningCurveResults Results;
    
    for(int p = 0; p < numPoints; ++p)
    {
        if(p == numPoints - 1)
        {
            m = numExamples;
        }
        
        vector<vector<double>> Xp(m,vector<double>(numFeatures,0));
        
        if(p == numPoints - 1)
        {
            Xp = X;
        }
        else
        {
            for(int i = 0; i < m; ++i)
            {
                Xp[i] = X[i];
            }
        }
        
        m += numIt;
        
        // K-fold cross validation
        KFoldResults KResults = KFoldCrossValidation(Xp, Y, numFolds);
        
        // Record mean and std for parameter value
        TrainMeanPerformance[p] = KResults.TrainMeanPerformance;
        TrainStdPerformance[p] = KResults.TrainStdPerformance;
        
        TestMeanPerformance[p] = KResults.TestMeanPerformance;
        TestStdPerformance[p] = KResults.TestStdPerformance;
    }
    
    // Return means and stds
    Results.TrainMeanPerformance = TrainMeanPerformance;
    Results.TrainStdPerformance = TrainStdPerformance;
    
    Results.TestMeanPerformance = TestMeanPerformance;
    Results.TestStdPerformance = TestStdPerformance;
    
    return Results;
}