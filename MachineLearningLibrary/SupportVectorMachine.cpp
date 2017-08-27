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
    MatrixXd F = Utilities::ConvertToEigen(X);
    MatrixXd YTrain = Utilities::ConvertToEigen(Y);
    
    // Set member variables
    numTrainExamples = X.size();
    numFeatures = X[0].size() + 1;
    
    ProcessFeatures(F, true);
    
    
    // Zero initialise the weights
    Theta = VectorXd::Zero(numFeatures);
    
    
    // Run Gradient Descent
    GradientDescent(XTrain, YTrain);
    
    return;
}

void SupportVectorMachine::ProcessFeatures(MatrixXd& F, bool bTraining)
{
    
    switch(Ker)
    {
        case Linear:
        {
            // Add a column of ones
            F.conservativeResize(F.rows(), F.cols()+1);
            F.col(F.cols()-1) = VectorXd::Ones(F.rows());
            
            break;
        }
        case Gaussian:
        {
            if(bTraining)
            {
                numFeatures = numTrainExamples + 1;
                XTrain = F;
            }
            
            F = GaussianKernel(F);
            
            break;
        }
    }
};


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
    MatrixXd F = Utilities::ConvertToEigen(X);
    
    ProcessFeatures(F, false);
    VectorXd Outputs = F*Theta;
    
    for(int i = 0; i < Outputs.size();  ++i)
    {
        Outputs(i) < 0 ? Predictions[i] = 0 : Predictions[i] = 1;
    }
    
    return Predictions;
}

MatrixXd SupportVectorMachine::GaussianKernel(const MatrixXd& X)
{
    
    MatrixXd F = MatrixXd::Zero(X.rows(), numTrainExamples);
    
    for(int example = 0; example < X.rows(); ++example)
    {
        MatrixXd Xi = (X.row(example)).replicate(numTrainExamples,1);
        
        VectorXd exponent = -(Xi - XTrain) * ((Xi - XTrain).transpose());
        exponent /= (2*Var);
        
        F.row(example) = exponent.array().exp();
        
    }
    
    // Add a column of ones
    F.conservativeResize(F.rows(), F.cols()+1);
    F.col(F.cols()-1) = VectorXd::Ones(F.rows());
    
    return F;
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






