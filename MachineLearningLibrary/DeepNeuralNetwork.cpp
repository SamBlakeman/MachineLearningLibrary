//
//  DeepNeuralNetwork.cpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 28/07/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#include "DeepNeuralNetwork.hpp"

DenseLayer::DenseLayer(int NumberOfUnits, int NumberOfInputs)
{
    numUnits = NumberOfUnits;
    numInputs = NumberOfInputs;
    
    w = MatrixXd::Zero(NumberOfUnits, NumberOfInputs+1);
    return;
}

DenseLayer::DenseLayer(int NumberOfUnits, int NumberOfInputs, ActivationFunction Activation)
{
    numUnits = NumberOfUnits;
    numInputs = NumberOfInputs;
    ActFun = Activation;
    
    w = MatrixXd::Zero(NumberOfUnits, NumberOfInputs+1);
    return;
}

///////////////////////////////////////////////////////////////////////

DeepNeuralNetwork::DeepNeuralNetwork(double alpha, double lambda, int numOutput, int Iters)
{
    Alpha = alpha;
    Lambda = lambda;
    numOut = numOutput;
    Iterations = Iters;
    
    return;
}

void DeepNeuralNetwork::AddDenseLayer(int NumberOfUnits, int NumberOfInputs)
{
    DenseLayer Layer(NumberOfUnits, NumberOfInputs);
    HiddenLayers.push_back(Layer);
}