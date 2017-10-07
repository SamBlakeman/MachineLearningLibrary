//
//  DeepAutoEncoder.cpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 06/10/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#include "DeepAutoEncoder.hpp"
#include "Utilities.hpp"
#include <iostream>

DeepAutoEncoder::DeepAutoEncoder(double alpha, double lambda, int Iters)
{
    Alpha = alpha;
    Lambda = lambda;
    Iterations = Iters;
    Costs = vector<double>(Iterations,0);
    
    return;
}

void DeepAutoEncoder::Fit(const vector<vector<double>>& X)
{
    MatrixXd XTrain = Utilities::ConvertToEigen(X);
    
    numFeatures = XTrain.cols();
    numTrainExamples = XTrain.rows();
    
    numOut = numFeatures + 1;
    
    // Add a column of ones
    XTrain.conservativeResize(XTrain.rows(), XTrain.cols()+1);
    XTrain.col(XTrain.cols()-1) = VectorXd::Ones(XTrain.rows());
    
    MatrixXd YTrain = XTrain;
    
    // Randomise output weights
    InitialiseHiddenWeights();
    InitialiseOutputWeights();
    
    cout << "Starting training\n";
    
    for(int iter=0; iter < Iterations; ++iter)
    {
        // Forward propagation
        MatrixXd Outputs = ForwardPropagation(XTrain);
        
        // Calculate Cost
        SumOfSquaredErrorsCosts(Outputs, YTrain, iter);
        
        // Partial derivatives
        vector<MatrixXd> Gradients = CalculateGradients(Outputs, XTrain, YTrain);
        
        // Update the weights
        UpdateLayers(Gradients);
        
    }
    
    return;
}

vector<vector<double>> DeepAutoEncoder::GetEncodedLayer(const vector<vector<double>>& X, int LayerToReadOut)
{
    if(LayerToReadOut <= 0)
    {
        cout << "Error - Layer 0 is the input layer";
        return vector<vector<double>> (0, vector<double> (0, 0));
    }
    else if(LayerToReadOut > HiddenLayers.size())
    {
        cout << "Error - Layer " << LayerToReadOut << " is not a hidden layer";
        return vector<vector<double>> (0, vector<double> (0, 0));
    }
    else
    {
        LayerToReadOut -= 1;
    }
    
    int numUnits = HiddenLayers[LayerToReadOut].GetNumberOfUnits();
    vector<vector<double>> EncodedLayer(numTrainExamples, vector<double>(numUnits, 0));
    
    
    // Foward propogate until the specified layer and record the activations for each example
    
    MatrixXd F = Utilities::ConvertToEigen(X);
    
    // Check for weights
    if(OutputWeights.isZero())
    {
        cout << endl << "Error in Predict() - No weights have been fit" << endl;
        return vector<vector<double>> (0, vector<double> (0, 0));
        
    }
    
    // Add a column of ones
    F.conservativeResize(F.rows(), F.cols()+1);
    F.col(F.cols()-1) = VectorXd::Ones(F.rows());
    
    
    MatrixXd NetInput;
    MatrixXd Activations = F;
    

    for(int l=0; l < HiddenLayers.size(); ++l)
    {
        DenseLayer* Layer = &HiddenLayers[l];
        Layer->Propagate(Activations);
        
        if(l == LayerToReadOut)
        {
            EncodedLayer = Utilities::ConvertFromEigen(Layer->GetActivations());
            break;
        }
        
    }
    
    return EncodedLayer;
}
