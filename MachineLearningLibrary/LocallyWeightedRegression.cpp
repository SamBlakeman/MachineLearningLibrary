//
//  LocallyWeightedRegression.cpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 23/08/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#include "LocallyWeightedRegression.hpp"
#include "Utilities.hpp"
#include <cmath>
#include <iostream>
#include <numeric>


LocallyWeightedRegression::LocallyWeightedRegression(double tau)
{
    Tau = tau;
}


void LocallyWeightedRegression::Fit(const vector<vector<double>>& XTrain, const vector<double>& YTrain)
{
    X = Utilities::ConvertToEigen(XTrain);
    
    // add intercept terms
    X.conservativeResize(X.rows(), X.cols()+1);
    X.col(X.cols()-1) = VectorXd::Ones(X.rows());
    
    Y = Utilities::ConvertToEigen(YTrain);
    
    numFeatures = int(XTrain[0].size());
    numTrainExamples = int(XTrain.size());
}


vector<double> LocallyWeightedRegression::Predict(const vector<vector<double>>& XTest)
{
    
    vector<double> Predictions(XTest.size(),0);
    
    // convert to Eigen
    MatrixXd XT = Utilities::ConvertToEigen(XTest);
    
    // add intercept terms
    XT.conservativeResize(XT.rows(), XT.cols()+1);
    XT.col(XT.cols()-1) = VectorXd::Ones(XT.rows());

    
    // loop through each test example
    for(int TestEx = 0; TestEx < XTest.size(); ++ TestEx)
    {
        VectorXd Xt = XT.row(TestEx);
        
        // calculate the weight distance matrix
        MatrixXd W(numTrainExamples, numTrainExamples);
        double distance = 0;
        
        for(int TrainEx = 0; TrainEx < numTrainExamples; ++ TrainEx)
        {
            VectorXd Xi = X.row(TrainEx);
            
            distance = (((Xi - Xt).transpose())*(Xi - Xt));
            distance = (-distance/(2*(pow(Tau, 2))));
            W(TrainEx, TrainEx) = exp(distance);
        }
        
        // solve the normal equation
        VectorXd Theta = (((X.transpose())*W*X).inverse())*(X.transpose())*W*Y;
    
        // use theta solution to make prediction
        Predictions[TestEx] = Xt.dot(Theta);
    }
    
    return Predictions;
    
}


double LocallyWeightedRegression::CalculateRSquared(const vector<vector<double>>& X, const vector<double>& Y)
{
    double RSquared = 0;
    
    // Check for weights
    if(X.empty())
    {
        cout << endl << "Error in CalculateRSquared() - Model has not been fit" << endl;
        return RSquared;
        
    }
    
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
