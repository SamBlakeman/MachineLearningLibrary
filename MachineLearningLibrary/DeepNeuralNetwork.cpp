//
//  DeepNeuralNetwork.cpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 28/07/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#include "DeepNeuralNetwork.hpp"
#include <iostream>

DenseLayer::DenseLayer(int NumberOfUnits, int NumberOfInputs)
{
    numUnits = NumberOfUnits;
    numInputs = NumberOfInputs;
    
    w = MatrixXd::Random(NumberOfUnits, NumberOfInputs+1);
    w /= 1000;
    return;
}

DenseLayer::DenseLayer(int NumberOfUnits, int NumberOfInputs, ActivationFunction Activation)
{
    numUnits = NumberOfUnits;
    numInputs = NumberOfInputs;
    ActFun = Activation;
    
    w = MatrixXd::Random(NumberOfUnits, NumberOfInputs+1);
    w /= 1000;
    return;
}

int DenseLayer::GetNumberOfUnits()
{
    return int(numUnits);
}

void DenseLayer::Propagate(MatrixXd& Inputs)
{
    Inputs = Inputs * w.transpose();
    Activate(Inputs);
    
    return;
}

void DenseLayer::Activate(MatrixXd& Mat)
{
    switch(ActFun)
    {
        case sigmoid:
        {
            
        }
        case relu:
        {
            
        }
        case linear:
        {
            
        }  
    }
    
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


void DeepNeuralNetwork::Fit(const vector<vector<double>>& X, const vector<vector<double>>& Y)
{
    pair<MatrixXd,MatrixXd> Eigens = ConvertToEigen(X, Y);
    MatrixXd XTrain = Eigens.first;
    MatrixXd YTrain = Eigens.second;
    
    numFeatures = XTrain.cols();
    numTrainExamples = XTrain.rows();
    
    // Add a column of ones
    XTrain.conservativeResize(XTrain.rows(), XTrain.cols()+1);
    XTrain.col(XTrain.cols()-1) = VectorXd::Ones(XTrain.rows());
    
    // Randomise output weights
    InitialiseOutputWeights();
    
    cout << "Starting training\n";
    
    for(int iter=0; iter < Iterations; ++iter)
    {
        // Forward propagation
        MatrixXd Outputs = ForwardPropagation(XTrain);
        
        // Calculate Cost
        //CalculateCosts(Outputs, YTrain, iter);
        
        // Partial derivatives
        //pair<MatrixXd,MatrixXd> Gradients = CalculateGradients(Outputs, XTrain, YTrain);
        
        // Update the weights
        //MatrixXd deltaW1 = Alpha * Gradients.first;
        //MatrixXd deltaW2 = Alpha * Gradients.second;
        
        //w1 = w1 - deltaW1;
        //w2 = w2 - deltaW2;
        
    }
    
    return;
}


void DeepNeuralNetwork::InitialiseOutputWeights()
{
    // Zero initialisation
    if(HiddenLayers.empty())
    {
        OutputWeights = MatrixXd::Random(numOut, numFeatures+1);
    }
    else
    {
        DenseLayer LastLayer = HiddenLayers[HiddenLayers.size()-1];
        OutputWeights = MatrixXd::Random(numOut, LastLayer.GetNumberOfUnits()+1);
    }
    
    OutputWeights /= 1000;
    
    return;
}


MatrixXd DeepNeuralNetwork::ForwardPropagation(const MatrixXd& X)
{
    MatrixXd NetInput;
    MatrixXd Activations = X;
    
    if(HiddenLayers.empty())
    {
        NetInput = Activations * OutputWeights.transpose();
    }
    else
    {
        for(int l=0; l < HiddenLayers.size(); ++l)
        {
            DenseLayer Layer = HiddenLayers[l];
            Layer.Propagate(Activations);
            
            // Add a column of ones (CHECK that we are always adding to the right dimension
            Activations.conservativeResize(Activations.rows(), Activations.cols()+1);
            Activations.col(Activations.cols()-1) = VectorXd::Ones(Activations.rows());
        }
        
        NetInput = Activations * OutputWeights.transpose();
    }
    
    Sigmoid(NetInput);
    
    return NetInput;
}


void DeepNeuralNetwork::Sigmoid(MatrixXd& Mat)
{
    MatrixXd Numerator = MatrixXd::Ones(Mat.rows(), Mat.cols());
    MatrixXd temp = -Mat;
    temp = temp.array().exp();
    MatrixXd Denominator = (MatrixXd::Ones(Mat.rows(), Mat.cols()) + temp);
    
    Mat = Numerator.array() / Denominator.array();
    
    return;
}


pair<MatrixXd,MatrixXd> DeepNeuralNetwork::ConvertToEigen(const vector<vector<double>>& XTrain, const vector<vector<double>>& YTrain )
{
    
    MatrixXd XT(XTrain.size(),XTrain[0].size());
    
    for(int i = 0; i < XTrain.size(); ++i)
    {
        vector<double> vec = XTrain[i];
        Eigen::VectorXd Xvec = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(vec.data(), vec.size());
        
        XT.row(i) = Xvec;
    }
    
    MatrixXd YT(YTrain.size(),YTrain[0].size());
    
    for(int i = 0; i < YTrain.size(); ++i)
    {
        vector<double> vec = YTrain[i];
        Eigen::VectorXd Yvec = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(vec.data(), vec.size());
        
        YT.row(i) = Yvec;
    }
    
    return make_pair(XT, YT);
}
