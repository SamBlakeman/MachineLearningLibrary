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
    double r = 4*sqrt(6/(numInputs+1+numOutputs));
    w = MatrixXd::Random(numUnits, numInputs+1)*r;
    //w = w.cwiseAbs();
    //w *= (numInputs+1);
    //w *= sqrt(2/(numInputs+1));
    
    return;
}

int DenseLayer::GetNumberOfUnits() const
{
    return int(numUnits);
}

void DenseLayer::Propagate(MatrixXd& Inputs)
{
    Inputs = Inputs * w.transpose();
    Activate(Inputs);
    
    // Add a column of ones to A for when we need to calculate the partial derivatives
    Inputs.conservativeResize(Inputs.rows(), Inputs.cols()+1);
    Inputs.col(Inputs.cols()-1) = VectorXd::Ones(Inputs.rows());
    A = Inputs;
    
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
        case leakyrelu:
        {
            LeakyReLU(Mat);
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

void DenseLayer::LeakyReLU(MatrixXd& Mat)
{
    for(int c=0; c < Mat.cols(); ++c)
    {
        for(int r=0; r < Mat.rows(); ++r)
        {
            if(Mat(r,c) < 0)
            {
                Mat(r,c) = 0.5 * Mat(r,c);
            }
        }
    }
    return;
}


double DenseLayer::GetPenalty() const
{
    return ((w.block(0, 0, w.rows(), w.cols()-1)).cwiseProduct((w.block(0, 0, w.rows(), w.cols()-1)))).sum();
}

MatrixXd DenseLayer::GetWeights() const
{
    return w;
}

MatrixXd DenseLayer::GetActivations() const
{
    return A;
}

MatrixXd DenseLayer::GetHiddenActivationGradient()
{
    switch(ActFun)
    {
        case sigmoid:
        {
            return (MatrixXd::Ones(A.rows(), A.cols()) - A).cwiseProduct(A);
        }
        case relu:
        {
            MatrixXd Gradients = MatrixXd::Zero(A.rows(), A.cols());
            for(int c=0; c < A.cols(); ++c)
            {
                for(int r=0; r < A.rows(); ++r)
                {
                    if(A(r,c) > 0)
                    {
                        Gradients(r,c) = 1;
                    }
                }
            }
            return Gradients;
        }
        case leakyrelu:
        {
            MatrixXd Gradients = MatrixXd::Zero(A.rows(), A.cols());
            Gradients = Gradients.array() + 0.5;
            for(int c=0; c < A.cols(); ++c)
            {
                for(int r=0; r < A.rows(); ++r)
                {
                    if(A(r,c) > 0)
                    {
                        Gradients(r,c) = 1;
                    }
                }
            }
            return Gradients;
        }
        case linear:
        {
            return MatrixXd::Ones(A.rows(), A.cols());
        }
    }
}

void DenseLayer::UpdateWeights(const MatrixXd& Grad, double Alpha)
{
    MatrixXd DeltaW = Alpha * Grad;
    w -= DeltaW;
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

void DeepNeuralNetwork::AddDenseLayer(int NumberOfUnits, int NumberOfInputs, ActivationFunction ActFun)
{
    DenseLayer Layer(NumberOfUnits, NumberOfInputs, ActFun);
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
        vector<MatrixXd> Gradients = CalculateGradients(Outputs, XTrain, YTrain);
        
        // Update the weights
        UpdateLayers(Gradients);
        
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
    
    if(iter%50 == 0)
    {
        cout << "Cost for iter " << iter << " = " << Costs[iter] << endl;
    }
    
    return;
}



vector<MatrixXd> DeepNeuralNetwork::CalculateGradients(const MatrixXd& Outputs, const MatrixXd& XTrain, const MatrixXd& YTrain)
{
    vector<MatrixXd> Deltas(HiddenLayers.size()+1);
    vector<MatrixXd> Grads(HiddenLayers.size()+1);
    
    // Get the output errors
    Deltas[HiddenLayers.size()] = Outputs - YTrain;
    Grads[HiddenLayers.size()] = Deltas[HiddenLayers.size()].transpose() * HiddenLayers[HiddenLayers.size()-1].GetActivations();
    
    for(int l = int(HiddenLayers.size()-1); l >= 0; --l)
    {
        if(l == int(HiddenLayers.size()-1))
        {
            Deltas[l] = Deltas[l+1] * OutputWeights;
        }
        else
        {
            Deltas[l] = Deltas[l+1].transpose() * HiddenLayers[l+1].GetWeights();
        }
        
        MatrixXd A = HiddenLayers[l].GetActivations();
        Deltas[l] = (HiddenLayers[l].GetHiddenActivationGradient()).cwiseProduct(Deltas[l]);
        Deltas[l].transposeInPlace();
        Deltas[l].conservativeResize(Deltas[l].rows()-1, Deltas[l].cols());
        
        if(l == 0)
        {
            Grads[l] = Deltas[l] * XTrain;
        }
        else
        {
            Grads[l] = Deltas[l] * HiddenLayers[l-1].GetActivations();
        }
        
        Grads[l] *= 1/numTrainExamples;
        Grads[l].block(0, 0, Grads[l].rows(), Grads[l].cols()-1) += (HiddenLayers[l].GetWeights().block(0, 0, HiddenLayers[l].GetWeights().rows(), HiddenLayers[l].GetWeights().cols()-1)) * (Lambda/numTrainExamples);
    }
    
    return Grads;
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


MatrixXd DeepNeuralNetwork::ConvertToEigen(const vector<vector<double>>& X)
{
    
    MatrixXd XT(X.size(),X[0].size());
    
    for(int i = 0; i < X.size(); ++i)
    {
        vector<double> vec = X[i];
        Eigen::VectorXd Xvec = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(vec.data(), vec.size());
        
        XT.row(i) = Xvec;
    }
    
    return XT;
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

void DeepNeuralNetwork::UpdateLayers(const vector<MatrixXd>& Grads)
{
    // Update Output Layer
    MatrixXd DeltaW = Alpha * Grads[HiddenLayers.size()];
    OutputWeights -= DeltaW;
    
    // Update Hidden Layers
    for(int l = 0; l < HiddenLayers.size(); ++l)
    {
        HiddenLayers[l].UpdateWeights(Grads[l], Alpha);
    }
    
}

vector<int> DeepNeuralNetwork::Predict(const vector<vector<double>>& XTest)
{
    MatrixXd X = ConvertToEigen(XTest);
    
    // Check for weights
    if(OutputWeights.isZero())
    {
        cout << endl << "Error in Predict() - No weights have been fit" << endl;
        return vector<int> (X.rows(),0);
        
    }
    
    // Add a column of ones
    X.conservativeResize(X.rows(), X.cols()+1);
    X.col(X.cols()-1) = VectorXd::Ones(X.rows());
    
    // Forward propagation
    MatrixXd Outputs = ForwardPropagation(X);
    
    // Get max prediction
    vector<int> Predictions = WinningOutput(Outputs);
    
    return Predictions;
}

vector<int> DeepNeuralNetwork::WinningOutput(const MatrixXd& Outputs)
{
    vector<int> Predictions(Outputs.rows(),0);
    
    for(int i=0; i < Outputs.rows(); ++i)
    {
        Outputs.row(i).maxCoeff( &Predictions[i] );
    }
    
    return Predictions;
}


double DeepNeuralNetwork::GetAccuracy(const vector<vector<double>>& X, const vector<double>& Y)
{
    double Accuracy;
    vector<int> Predictions = Predict(X);
    
    int numCorrect = 0;
    
    for(int i = 0; i < X.size(); ++i)
    {
        if(Y[i] == Predictions[i])
        {
            ++numCorrect;
        }
    }
    
    Accuracy = (double(numCorrect)/X.size())*100;
    
    return Accuracy;
}
