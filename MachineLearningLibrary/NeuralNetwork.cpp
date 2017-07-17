//
//  NeuralNetwork.cpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 06/06/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#include "NeuralNetwork.hpp"
#include "Utilities.hpp"
#include <stdlib.h>
#include <cmath>
#include <iostream>
#include <numeric>


NeuralNetwork::NeuralNetwork(double alpha, double lambda, int numHidden, int numOutput, int Iters)
{
    Alpha = alpha;
    Lambda = lambda;
    numHid = numHidden;
    numOut = numOutput;
    Iterations = Iters;
}



void NeuralNetwork::Fit(MatrixXd XTrain, const MatrixXd& YTrain)
{
    numFeatures = XTrain.cols();
    numTrainExamples = XTrain.rows();
    
    // Add a column of ones
    XTrain.conservativeResize(XTrain.rows(), XTrain.cols()+1);
    XTrain.col(XTrain.cols()-1) = MatrixXd::Ones(XTrain.rows());
    
    // Randomise weights to start with
    InitialiseWeights();
    
    for(int iter=0; iter < Iterations; ++iter)
    {
    
        // Forward propagation
        MatrixXd Outputs = ForwardPropagation(XTrain);
    
        // Calculate Cost
        CalculateCosts(Outputs, YTrain, iter);
        
        // Partial derivatives
        pair<MatrixXd,MatrixXd> Gradients = CalculateGradients(Outputs, XTrain, YTrain);
        
        // Update the weights
        MatrixXd deltaW1 = Alpha * Gradients.first;
        MatrixXd deltaW2 = Alpha * Gradients.second;
        
        w1 = w1 - deltaW1;
        w2 = w2 - deltaW2;
        
    }
    
    
    return;
}


void NeuralNetwork::InitialiseWeights()
{
    // Zero initialisation
    w1.Random(numHid, numFeatures+1);
    w2.Random(numOut, numHid+1);
    
    w1 /= 100;
    w2 /= 100;
    
    return;
}


void NeuralNetwork::Sigmoid(vector<double>& Vec)
{
    for(int i = 0; i < Vec.size(); ++i)
    {
        Vec[i] = 1/(1+exp(-Vec[i]));
    }
    
    return;
}

void NeuralNetwork::Sigmoid(MatrixXd& Mat)
{
    for(int r = 0; r < Mat.rows(); ++r)
    {
        for(int c = 0; c < Mat.cols(); ++c)
        {
            Mat(r,c) = 1/(1+exp(-Mat(r,c)));
        }
    }
    
    return;
}




VectorXd NeuralNetwork::Predict(MatrixXd XTest)
{
    MatrixXd Outputs;
    
    // Check for weights
    if(w1.isZero() || w2.isZero())
    {
        cout << endl << "Error in Predict() - No weights have been fit" << endl;
        return VectorXd (XTest.rows(),0);
        
    }
    
    // Add a column of ones
    XTest.conservativeResize(XTest.rows(), XTest.cols()+1);
    XTest.col(XTest.cols()-1) = MatrixXd::Ones(XTest.rows());
    
    // Forward propagation
    Outputs = ForwardPropagation(XTest);
    
    // Get max prediction
    VectorXd Predictions = WinningOutput(Outputs);
    
    return Predictions;
}


MatrixXd NeuralNetwork::ForwardPropagation(const MatrixXd& X)
{
    MatrixXd a2 = X * w1.transpose();
    Sigmoid(a2);
    
    // Add a column of ones
    a2.conservativeResize(a2.rows(), a2.cols()+1);
    a2.col(a2.cols()-1) = MatrixXd::Ones(a2.rows());
    
    MatrixXd a3 = a2 * w2.transpose();
    Sigmoid(a3);
    
    return a3;
}


VectorXd NeuralNetwork::WinningOutput(MatrixXd Outputs)
{
    VectorXd Predictions;
    
    for(int i=0; i < Outputs.size(); ++i)
    {
        vector<double>::iterator max = max_element(Outputs[i].begin(), Outputs[i].end());
        Predictions[i] = distance(Outputs[i].begin(), max);
    }
    
    return Predictions;
}


void NeuralNetwork::CalculateCosts(const MatrixXd& Outputs, const MatrixXd& YTrain, int iter)
{
    VectorXd UnitCosts(numOut,0); // check it creates a vector of 0's
    
    for(int i = 0; i < numTrainExamples; ++i) // TODO vectorise
    {
        
        
        VectorXd logOutputs = Outputs.row(i);
        
        for(int l = 0; l < logOutputs.size(); ++l)
        {
            logOutputs(l) = log(logOutputs(l));
        }
        
        VectorXd logOneMinusOutputs = Outputs.row(i);
        for(int l = 0; l < logOneMinusOutputs.size(); ++l)
        {
            logOneMinusOutputs(l) = log(1 - logOneMinusOutputs(l));
        }
        
        
        VectorXd term1  = (YTrain.row(i) * -1) * logOutputs;
        VectorXd term2 = (VectorXd::Ones(YTrain.cols()) - YTrain.row(i)) * logOneMinusOutputs;
        
        UnitCosts = UnitCosts + (term1 - term2);
        
    }
    
    Costs.push_back(0);
    Costs[iter] = UnitCosts.sum();
    Costs[iter] *= (1/numTrainExamples);
    
    double RegTerm = 0;
    
    for (int h = 0; h < numHid; ++h)
    {
        for(int f = 0; f < numFeatures; ++f)
        {
            RegTerm += pow(w1(h,f), 2);
        }
    }
    
    for (int o = 0; o < numOut; ++o)
    {
        for(int h = 0; h < numHid; ++h)
        {
            RegTerm += pow(w2(o,h), 2);
        }
    }
    
    
    RegTerm *= Lambda/(2*numTrainExamples);
    Costs[iter] += RegTerm;
    
    return;
}


pair<MatrixXd,MatrixXd> NeuralNetwork::CalculateGradients(const MatrixXd& Outputs, const MatrixXd& XTrain, const MatrixXd& YTrain)
{
    // Get the output errors
    MatrixXd delta3 = Outputs - YTrain;
    
    // Get the net input into layer 2
    MatrixXd z2 = XTrain * w1.transpose();
    
    // Add a column of ones
    z2.conservativeResize(z2.rows(), z2.cols()+1);
    z2.col(z2.cols()-1) = MatrixXd::Ones(z2.rows());
    
    // Get the layer 2 errors
    MatrixXd delta2 = delta3 * w2;
    Sigmoid(z2);
    delta2 = delta2 *(z2 * (MatrixXd::Ones(z2.rows(), z2.cols()) - z2));
    delta2 = delta2.transpose();
    delta2.erase(delta2.end() - 1);
    
    // Calculate the two gradients
    MatrixXd grad1 = delta2 * XTrain;
    MatrixXd grad2 = delta3.transpose() * z2;
    
    // Regularise
    grad1 += w1 * Lambda;
    grad2 += w2 * Lambda;
    
    return make_pair(grad1, grad2);
}


/*void NeuralNetwork::AddBiasUnit(MatrixXd &Activations, BiasLocation Location)
{
    if(Location == Row)
    {
        // Add a column of ones
        for(int i = 0; i < Activations.size(); ++i)
        {
            Activations[i].push_back(1);
        }
    }
    else if(Location == Column)
    {
        // Add a row of ones
        Activations.push_back(vector<double> (Activations[0].size(), 1));
    }
    
    return;
}*/

vector<double> NeuralNetwork::GetCosts()
{
    // Check for weights
    if(Costs.empty())
    {
        cout << endl << "Error in GetCosts() - No costs have been calculated" << endl;
    }
    
    return Costs;
}
