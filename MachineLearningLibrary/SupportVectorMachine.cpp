//
//  SupportVectorMachine.cpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 25/08/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#include "SupportVectorMachine.hpp"
#include "Utilities.hpp"


SupportVectorMachine::SupportVectorMachine(double c, double alpha, int iters)
{
    C = c;
    Alpha = alpha;
    Iterations = iters;
    Costs = vector<double>(Iterations, 0);
}


void SupportVectorMachine::AddGaussianKernel(double Variance)
{
    Var = Variance;
    Ker = Gaussian;
    return;
}


void SupportVectorMachine::Fit(const vector<vector<double>>& X, const vector<double>& Y)
{
    // Convert to Eigen
    MatrixXd XTrain = Utilities::ConvertToEigen(X);
    MatrixXd YTrain = Utilities::ConvertToEigen(Y);
    
    // Add a column of ones
    XTrain.conservativeResize(XTrain.rows(), XTrain.cols()+1);
    XTrain.col(XTrain.cols()-1) = VectorXd::Ones(XTrain.rows());
    
    // Set member variables
    numTrainExamples = X.size();
    
    switch(Ker)
    {
        case Linear:
        {
            numFeatures = X[0].size() + 1;
            break;
        }
        case Gaussian:
        {
            numFeatures = numTrainExamples;
            break;
        }
    }
    
    // Zero initialise the weights
    Theta = VectorXd::Zero(numFeatures);
    
    
    // Run Gradient Descent
    GradientDescent(XTrain, YTrain);
    
    return;
}


void SupportVectorMachine::GradientDescent(const MatrixXd& XTrain, const MatrixXd& YTrain)
{
    for(int iter = 0; iter < Iterations; ++iter)
    {
        
        // Predictions
        
        
        // Cost
        
        
        // Partial derivatives
        
        
        // Update theta
        
        
        
    }
    
    return;
}



vector<double> SupportVectorMachine::Predict(const vector<vector<double>>& X)
{
    vector<double> Predictions(X.size(),0);
    MatrixXd F;
    
    switch(Ker)
    {
        case Linear:
        {
            F = Utilities::ConvertToEigen(X);
            break;
        }
        case Gaussian:
        {
            
            break;
        }
    }
    
    
    VectorXd Outputs = F*Theta;
    
    return Predictions;
}


vector<double> SupportVectorMachine::GetCosts()
{
    return Costs;
}


double SupportVectorMachine::GetAccuracy(const vector<vector<double>>& X, const vector<double>& Y)
{
    double Accuracy = 0;
    
    
    return Accuracy;
}






