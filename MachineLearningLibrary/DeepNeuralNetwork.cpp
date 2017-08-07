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
    
    return;
}

DenseLayer::DenseLayer(int NumberOfUnits, int NumberOfInputs, ActivationFunction Activation)
{
    numUnits = NumberOfUnits;
    numInputs = NumberOfInputs;
    ActFun = Activation;
    
    return;
}

void DenseLayer::InitialiseWeights(int NumberOfOutputs)
{
    numOutputs = NumberOfOutputs;
    
    // Xavier Initialization
    double r = sqrt(6/(numInputs+1+numOutputs));
    w = MatrixXd::Random(numUnits, numInputs+1)*r;
    
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
            Sigmoid(Mat);
            return;
        }
        case relu:
        {
            ReLU(Mat);
            return;
        }
        case linear:
        {
            Linear(Mat);
            return;
        }  
    }
    
}


void DenseLayer::Linear(MatrixXd& Mat)
{
    return;
}


void DenseLayer::Sigmoid(MatrixXd& Mat)
{
    MatrixXd Numerator = MatrixXd::Ones(Mat.rows(), Mat.cols());
    MatrixXd temp = -Mat;
    temp = temp.array().exp();
    MatrixXd Denominator = (MatrixXd::Ones(Mat.rows(), Mat.cols()) + temp);
    
    Mat = Numerator.array() / Denominator.array();
    
    return;
}


void DenseLayer::ReLU(MatrixXd& Mat)
{
    for(int c=0; c < Mat.cols(); ++c)
    {
        for(int r=0; r < Mat.rows(); ++r)
        {
            if(Mat(r,c) < 0)
            {
                Mat(r,c) = 0;
            }
        }
    }
    return;
}

double DenseLayer::GetPenalty()
{
    return ((w.block(0, 0, w.rows(), w.cols()-1)).cwiseProduct((w.block(0, 0, w.rows(), w.cols()-1)))).sum();
}

///////////////////////////////////////////////////////////////////////

DeepNeuralNetwork::DeepNeuralNetwork(double alpha, double lambda, int numOutput, int Iters)
{
    Alpha = alpha;
    Lambda = lambda;
    numOut = numOutput;
    Iterations = Iters;
    Costs = vector<double>(Iterations,0);
    
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
    InitialiseHiddenWeights();
    InitialiseOutputWeights();
    
    cout << "Starting training\n";
    
    for(int iter=0; iter < Iterations; ++iter)
    {
        // Forward propagation
        MatrixXd Outputs = ForwardPropagation(XTrain);
        
        // Calculate Cost
        CalculateCosts(Outputs, YTrain, iter);
        
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
        double r = sqrt(6/(numFeatures+1+1));
        OutputWeights = MatrixXd::Random(numOut, numFeatures+1)*r;
    }
    else
    {
        DenseLayer* LastLayer = &HiddenLayers[HiddenLayers.size()-1];
        double r = sqrt(6/(LastLayer->GetNumberOfUnits()+1+1));
        OutputWeights = MatrixXd::Random(numOut, LastLayer->GetNumberOfUnits()+1)*r;
    }
    
    return;
}

void DeepNeuralNetwork::InitialiseHiddenWeights()
{
    if(!HiddenLayers.empty())
    {
        for(int l=0; l < HiddenLayers.size(); ++l)
        {
            DenseLayer* Layer = &HiddenLayers[l];
            int NumberOfOutputs = HiddenLayers[l+1].GetNumberOfUnits();
            Layer->InitialiseWeights(NumberOfOutputs);
        }
    }
    
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
            DenseLayer* Layer = &HiddenLayers[l];
            Layer->Propagate(Activations);
            
            // Add a column of ones (CHECK that we are always adding to the right dimension)
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


void DeepNeuralNetwork::CalculateCosts(const MatrixXd& Outputs, const MatrixXd& YTrain, int iter)
{
    MatrixXd LogOutputs = Outputs.array().log();
    MatrixXd term1 = -(YTrain.cwiseProduct(LogOutputs));
    
    MatrixXd logOneMinusOutputs = (MatrixXd::Ones(Outputs.rows(), Outputs.cols()) - Outputs).array().log();
    MatrixXd term2 = (MatrixXd::Ones(YTrain.rows(),YTrain.cols()) - YTrain).cwiseProduct(logOneMinusOutputs);
    
    Costs[iter] = (term1 - term2).sum();
    Costs[iter] *= (1/numTrainExamples);
    
    double RegTerm = 0;
    
    if(!HiddenLayers.empty())
    {
        for(int l=0; l < HiddenLayers.size(); ++l)
        {
            RegTerm += HiddenLayers[l].GetPenalty();
        }
        
    }
    
    RegTerm += ((OutputWeights.block(0, 0, OutputWeights.rows(), OutputWeights.cols()-1)).cwiseProduct((OutputWeights.block(0, 0, OutputWeights.rows(), OutputWeights.cols()-1)))).sum();
    
    RegTerm *= Lambda/(2*numTrainExamples);
    Costs[iter] += RegTerm;
    
    cout << "Cost for iter " << iter << " = " << Costs[iter] << endl;
    
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


vector<double> DeepNeuralNetwork::GetCosts() const
{
    // Check for weights
    if(Costs.empty())
    {
        cout << endl << "Error in GetCosts() - No costs have been calculated" << endl;
    }
    
    return Costs;
}
