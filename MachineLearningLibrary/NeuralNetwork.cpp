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


NeuralNetwork::NeuralNetwork(double alpha, double lambda, int numHidden, int Iters)
{
    Alpha = alpha;
    Lambda = lambda;
    numHid = numHidden;
    Iterations = Iters;
}



void NeuralNetwork::Fit(vector<vector<double>> XTrain, const vector<vector<double>>& YTrain)
{
    numFeatures = XTrain[0].size();
    numTrainExamples = XTrain.size();
    
    // Add a column of ones
    for(int i = 0; i < XTrain.size(); ++i)
    {
        XTrain[i].push_back(1);
    }
    
    // Randomise weights to start with
    InitialiseWeights();
    
    for(int iter=0; iter < Iterations; ++iter)
    {
    
        // Forward propagation
        vector<vector<double>> Outputs = ForwardPropagation(XTrain);
    
        // Calculate Cost
        CalculateCosts(Outputs, YTrain, iter);
        
        
    }
    
    
    return;
}


void NeuralNetwork::InitialiseWeights()
{
    // Zero initialisation
    w1 = vector<vector<double>> (numHid, vector<double>(numFeatures + 1, 0));
    w2 = vector<vector<double>> (numOut, vector<double>(numHid + 1, 0));
    
    // Randomise the weights
    for(int r = 0; r < w1.size(); ++r)
    {
        for(int c = 0; c < w1[0].size(); ++c)
        {
            w1[r][c] = (rand() % 100)/1000;
        }
    }
    
    for(int r = 0; r < w2.size(); ++r)
    {
        for(int c = 0; c < w2[0].size(); ++c)
        {
            w2[r][c] = (rand() % 100)/1000;
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
    
    AddBiasUnit(a2);
    
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


void NeuralNetwork::CalculateCosts(const vector<vector<double>>& Outputs, const vector<vector<double>>& YTrain, int iter)
{
    vector<double> UnitCosts(numOut,0);
    
    for(int i = 0; i < numTrainExamples; ++i) // TODO vectorise
    {
        
        
        vector<double> logOutputs = Outputs[i];
        for(int l = 0; l < logOutputs.size(); ++l)
        {
            logOutputs[l] = log(logOutputs[l]);
        }
        
        vector<double> logOneMinusOutputs = Outputs[i];
        for(int l = 0; l < logOneMinusOutputs.size(); ++l)
        {
            logOneMinusOutputs[l] = log(1 - logOneMinusOutputs[l]);
        }
        
        
        vector<double> term1  = Utilities::ScalarMult(Utilities::ScalarMult(YTrain[i],-1), logOutputs);
        vector<double> term2 = Utilities::ScalarMult(Utilities::ScalarSub(1, YTrain[i]), logOneMinusOutputs) ;
        
        UnitCosts = Utilities::ScalarAdd(UnitCosts, Utilities::ScalarSub(term1, term2));
        
    }
    
    double start = 0;
    Costs[iter] = accumulate(UnitCosts.begin(), UnitCosts.end(), start);
    Costs[iter] *= (1/numTrainExamples);
    
    
    
    double RegTerm = 0;
    
    for (int h = 0; h < numHid; ++h)
    {
        for(int f = 0; f < numFeatures; ++f)
        {
            RegTerm += pow(w1[h][f], 2);
        }
    }
    
    for (int o = 0; o < numOut; ++o)
    {
        for(int h = 0; h < numHid; ++h)
        {
            RegTerm += pow(w2[o][h], 2);
        }
    }
    
    
    RegTerm *= Lambda/(2*numTrainExamples);
    Costs[iter] += RegTerm;
    
    return;
}


void NeuralNetwork::AddBiasUnit(vector<vector<double>> &Activations)
{
    Activations.push_back(vector<double>(Activations[0].size(),1));
    
    return;
}
