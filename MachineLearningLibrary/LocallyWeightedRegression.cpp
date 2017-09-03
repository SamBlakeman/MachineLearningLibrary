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
    CostFun = SumOfSquaredErrors;
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
