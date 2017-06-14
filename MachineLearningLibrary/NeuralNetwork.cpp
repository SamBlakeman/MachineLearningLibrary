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


NeuralNetwork::NeuralNetwork(double alpha, double lambda, int numHidden, int Iters)
{
    Alpha = alpha;
    Lambda = lambda;
    numHid = numHidden;
    Iterations = Iters;
}



void NeuralNetwork::Fit(vector<vector<double>> XTrain, const vector<double>& YTrain)
{
    numFeatures = XTrain[0].size() + 1;
    numTrainExamples = XTrain.size();
    
    // Add a column of ones
    for(int i = 0; i < XTrain.size(); ++i)
    {
        XTrain[i].push_back(1);
    }
    
    // Randomise weights to start with
    InitialiseWeights();
    
    
    
    
    return;
}


void NeuralNetwork::InitialiseWeights()
{
    // Zero initialisation
    w1 = vector<vector<double>> (numHid, vector<double>(numFeatures, 0));
    w2 = vector<vector<double>> (numOut, vector<double>(numHid, 0));
    
    // Randomise the weights
    for(int r = 0; r < w1.size(); ++r)
    {
        for(int c = 0; c < w1[0].size(); ++c)
        {
            w1[r][c] = (rand() % 100)/100;
        }
    }
    
    for(int r = 0; r < w2.size(); ++r)
    {
        for(int c = 0; c < w2[0].size(); ++c)
        {
            w2[r][c] = (rand() % 100)/100;
        }
    }
    
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

void NeuralNetwork::Sigmoid(vector<vector<double>>& Mat)
{
    for(int r = 0; r < Mat.size(); ++r)
    {
        for(int c = 0; c < Mat[0].size(); ++c)
        {
            Mat[r][c] = 1/(1+exp(-Mat[r][c]));
        }
    }
    
    return;
}




vector<double> NeuralNetwork::Predict(vector<vector<double>> XTest)
{
    vector<vector<double>> Outputs(XTest.size(), vector<double>(numOut,0));
    
    // Check for weights
    if(w1.empty() || w2.empty())
    {
        cout << endl << "Error in Predict() - No weights have been fit" << endl;
        return vector<double> (XTest.size(),0);
        
    }
    
    // Add a column of ones
    for(int i = 0; i < XTest.size(); ++i)
    {
        XTest[i].push_back(1);
    }
    
    // Forward propagation
    Outputs = ForwardPropagation(XTest);
    
    // Get max prediction
    vector<double> Predictions = WinningOutput(Outputs);
    
    return Predictions;
}


vector<vector<double>> NeuralNetwork::ForwardPropagation(const vector<vector<double>>& X)
{
    vector<vector<double>> a2 = Utilities::Product(w1,Utilities::Transpose(X));
    Sigmoid(a2);
    
    vector<vector<double>> a3 = Utilities::Product(w2, a2);
    Sigmoid(a3);
    
    return a3;
}


vector<double> NeuralNetwork::WinningOutput(vector<vector<double>> Outputs)
{
    vector<double> Predictions;
    
    for(int i=0; i < Outputs.size(); ++i)
    {
        vector<double>::iterator max = max_element(Outputs[i].begin(), Outputs[i].end());
        Predictions[i] = distance(Outputs[i].begin(), max);
    }
    
    return Predictions;
}
