//
//  DeepNeuralNetwork.cpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 28/07/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#include "DeepNeuralNetwork.hpp"
#include "Utilities.hpp"
#include <iostream>
#include <numeric>

DeepNeuralNetwork::DeepNeuralNetwork(double alpha, double lambda, int numOutput, int Iters, CostFunction Cost)
{
    Alpha = alpha;
    Lambda = lambda;
    numOut = numOutput;
    Iterations = Iters;
    Costs = vector<double>(Iterations,0);
    CostFun = Cost;
    
    return;
}


void DeepNeuralNetwork::Fit(const vector<vector<double>>& X, const vector<double>& Y)
{
    // One hot encode Y if neccessary
    vector<vector<double>> YEnc = OneHotEncode(Y);
    
    pair<MatrixXd,MatrixXd> Eigens = Utilities::ConvertToEigen(X, YEnc);
    MatrixXd XTrain = Eigens.first;
    MatrixXd YTrain = Eigens.second;
    
    numFeatures = XTrain.cols();
    numTrainExamples = XTrain.rows();
    
    // Add a column of ones
    XTrain.conservativeResize(XTrain.rows(), XTrain.cols()+1);
    XTrain.col(XTrain.cols()-1) = VectorXd::Ones(XTrain.rows());
    
    // Randomise output weights
    InitialiseHiddenWeights();
    InitialiseOutputWeights();
    
    cout << "Starting training\n";
    
    for(int iter=0; iter < Iterations; ++iter)
    {
        // Forward propagation
        MatrixXd Outputs = ForwardPropagation(XTrain);
        
        // Handle the activation function of the output units
        switch(CostFun)
        {
            case CrossEntropy:
            {
                Sigmoid(Outputs);
                break;
            }
            case SumOfSquaredErrors:
            {
                break;
            }
        }
        
        // Calculate Cost
        CalculateCosts(Outputs, YTrain, iter);
        
        // Partial derivatives
        vector<MatrixXd> Gradients = CalculateGradients(Outputs, XTrain, YTrain);
        
        // Update the weights
        UpdateLayers(Gradients);
        
    }
    
    return;
}


vector<vector<double>> DeepNeuralNetwork::OneHotEncode(const vector<double>& Y)
{
    int numExamples = (int)Y.size();
    vector<vector<double>> EncodedY (numExamples, vector<double>(numOut, 0));
    
    if(numOut == 1)
    {
        for(int e = 0; e < numExamples; ++e)
        {
            EncodedY[e][0] = Y[e];
        }
    }
    else
    {
        for(int e = 0; e < numExamples; ++e)
        {
            if(Y[e] >= EncodedY[0].size())
            {
                cout << "There are not enough output units to encode Y!" << endl;
            }
            else
            {
                EncodedY[e][Y[e]] = 1;
            }
        }
    }
    
    return EncodedY;
}


void DeepNeuralNetwork::CalculateCosts(const MatrixXd& Outputs, const MatrixXd& YTrain, const int& iter)
{
    switch(CostFun)
    {
        case CrossEntropy:
        {
            CrossEntropyCosts(Outputs, YTrain, iter);
            break;
        }
        case SumOfSquaredErrors:
        {
            SumOfSquaredErrorsCosts(Outputs, YTrain, iter);
            break;
        }
    }
    
    Regularize(iter);
    
    if(iter%50 == 0)
    {
        cout << "Cost for iter " << iter << " = " << Costs[iter] << endl;
    }
    
    return;
}


vector<double> DeepNeuralNetwork::Predict(const vector<vector<double>>& XTest)
{
    MatrixXd X = Utilities::ConvertToEigen(XTest);
    
    // Check for weights
    if(OutputWeights.isZero())
    {
        cout << endl << "Error in Predict() - No weights have been fit" << endl;
        return vector<double> (X.rows(),0);
        
    }
    
    // Add a column of ones
    X.conservativeResize(X.rows(), X.cols()+1);
    X.col(X.cols()-1) = VectorXd::Ones(X.rows());
    
    // Forward propagation
    MatrixXd Outputs = ForwardPropagation(X);
    
    // Handle the activation function of the output units
    switch(CostFun)
    {
        case CrossEntropy:
        {
            Sigmoid(Outputs);
            break;
        }
        case SumOfSquaredErrors:
        {
            break;
        }
    }
    
    vector<double> Predictions(Outputs.rows(),0);
    
    if(CostFun == CrossEntropy)
    {
        // Get max prediction
        Predictions = WinningOutput(Outputs);
    }
    else
    {
        for(int i = 0; i < Outputs.rows(); ++i)
        {
            Predictions[i] = Outputs(i,0);
        }
    }
    
    return Predictions;
}

vector<double> DeepNeuralNetwork::WinningOutput(const MatrixXd& Outputs)
{
    vector<double> Predictions(Outputs.rows(),0);
    
    for(int i=0; i < Outputs.rows(); ++i)
    {
        Outputs.row(i).maxCoeff( &Predictions[i] );
    }
    
    return Predictions;
}


void DeepNeuralNetwork::PreTrain(const vector<vector<double>>& X)
{
    
    
    
    return;
}

void DeepNeuralNetwork::SetLambda(double lambda)
{
    Lambda = lambda;
}

void DeepNeuralNetwork::SetAlpha(double alpha)
{
    Alpha = alpha;
}

void DeepNeuralNetwork::SetIterations(int iters)
{
    Iterations = iters;
}

void DeepNeuralNetwork::SetTau(double tau)
{
    
}

void DeepNeuralNetwork::SetC(double c)
{
    
}

void DeepNeuralNetwork::SetVar(double var)
{
    
}
