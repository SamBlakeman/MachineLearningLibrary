//
//  SupportVectorMachine.cpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 25/08/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#include "SupportVectorMachine.hpp"
#include "Utilities.hpp"
#include <iostream>


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
    GradientDescent(F, YTrain);
    
    return;
}

void SupportVectorMachine::ProcessFeatures(MatrixXd& F, bool bTraining)
{
    switch(Ker)
    {
        case Linear:
        {
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
    
    // Add a column of ones
    F.conservativeResize(F.rows(), F.cols()+1);
    F.col(F.cols()-1) = VectorXd::Ones(F.rows());
    
};


void SupportVectorMachine::GradientDescent(const MatrixXd& F, const MatrixXd& YTrain)
{
    for(int iter = 0; iter < Iterations; ++iter)
    {
        
        // Predictions
        VectorXd Outputs = F*Theta;
        
        // Cost
        CalculateCost(Outputs, YTrain, iter);
        
        // Partial derivatives
        VectorXd Gradients = CalculateGradients(Outputs, F, YTrain);
        
        // Update theta
        UpdateTheta(Gradients);
        
    }
    
    return;
}

void SupportVectorMachine::CalculateCost(const VectorXd& Outputs, const MatrixXd& YTrain, int iter)
{
    
    VectorXd Ones = VectorXd::Ones(YTrain.rows(), YTrain.cols());
    
    VectorXd ExampleCosts = YTrain.cwiseProduct(Cost1(Outputs)) + (Ones - YTrain).cwiseProduct(Cost0(Outputs));
    
    Costs[iter] = ExampleCosts.sum();
    Costs[iter] *= C;
    
    // Regularize
    double R = Theta.dot(Theta);
    Costs[iter] += (1/2)*R;
    
    if(iter%50 == 0)
    {
        cout << "Cost for iter " << iter << " = " << Costs[iter] << endl;
    }
    
    return;
}


VectorXd SupportVectorMachine::Cost1(const VectorXd& Outputs)
{
    VectorXd costs = Outputs.cwiseMin(1);
    costs *= -1;
    costs = costs.array() + 1;
    
    return costs;
}


VectorXd SupportVectorMachine::Cost0(const VectorXd& Outputs)
{
    VectorXd costs = Outputs.cwiseMax(-1);
    costs = costs.array() + 1;
    
    return costs;
}


VectorXd SupportVectorMachine::CalculateGradients(const MatrixXd& Outputs, const MatrixXd& F, const MatrixXd& YTrain)
{
    VectorXd Gradients = VectorXd::Zero(Theta.size());
    
    for(int i = 0; i < numTrainExamples; ++i)
    {
        if(YTrain(i) == 1)
        {
            if(Outputs(i) < 1)
            {
                Gradients += -(F.row(i));
            }
        }
        else
        {
            if(Outputs(i) > -1)
            {
                Gradients += F.row(i);
            }
        }
    }
    
    Gradients *= C;
    Gradients += Theta;
    Gradients(Gradients.size()-1) -= Theta(Theta.size()-1); // do not regularize the intercept term
    
    return Gradients;
}


void SupportVectorMachine::UpdateTheta(const VectorXd& Gradients)
{
    Theta -= Alpha*(Gradients);
    
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
    cout << "\nConstructing Gaussian Kernel...\n";
    
    MatrixXd F = MatrixXd::Zero(X.rows(), numTrainExamples);
    
    for(int example = 0; example < X.rows(); ++example)
    {
        MatrixXd Xi = (X.row(example)).replicate(numTrainExamples,1);
        
        MatrixXd Differences = Xi - XTrain;
        
        MatrixXd e = -(Xi - XTrain) * ((Xi - XTrain).transpose());
        VectorXd exponent = e.diagonal();
        exponent /= (2*Var);
        
        F.row(example) = exponent.array().exp();
        
    }
    
    return F;
}


vector<double> SupportVectorMachine::GetCosts()
{
    return Costs;
}


double SupportVectorMachine::GetAccuracy(const vector<vector<double>>& X, const vector<double>& Y)
{
    double Accuracy = 0;
    
    vector<double> Predictions = Predict(X);
    
    double numCorrect = 0;
    double total = Y.size();
    
    for(int e = 0; e < total; ++e)
    {
        if(Predictions[e] == Y[e])
        {
            ++numCorrect;
        }
    }
    
    Accuracy = (numCorrect/total)*100;
    
    
    return Accuracy;
}






