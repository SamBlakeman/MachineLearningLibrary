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

NeuralNetwork::NeuralNetwork(double alpha, double lambda, int numHidden, int numOutput, int Iters, CostFunction Cost)
{
    Alpha = alpha;
    Lambda = lambda;
    numHid = numHidden;
    numOut = numOutput;
    Iterations = Iters;
    Costs = vector<double>(Iterations,0);
    CostFun = Cost;
}

NeuralNetwork::NeuralNetwork(double alpha, double lambda, int numHidden, int numOutput, int Iters, CostFunction Cost, ActivationFunction HiddenActivation)
{
    Alpha = alpha;
    Lambda = lambda;
    numHid = numHidden;
    numOut = numOutput;
    Iterations = Iters;
    Costs = vector<double>(Iterations,0);
    HiddenActFun = HiddenActivation;
    CostFun = Cost;
}



void NeuralNetwork::Fit(const vector<vector<double>>& X, const vector<double>& Y)
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
    
    // Randomise weights to start with
    InitialiseWeights();
    
    cout << "Starting training\n";
    
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


vector<vector<double>> NeuralNetwork::OneHotEncode(const vector<double>& Y)
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


void NeuralNetwork::InitialiseWeights()
{
    // Xavier initialisation
    double r1 = sqrt(6/(numFeatures+1+numOut));
    double r2 = sqrt(6/(numHid+1+1));
    
    if(HiddenActFun == relu || HiddenActFun == leakyrelu)
    {
        r1 *= sqrt(2);
    }
    
    w1 = MatrixXd::Random(numHid, numFeatures+1)*r1;
    w2 = MatrixXd::Random(numOut, numHid+1)*r2;
    
    return;
}

void NeuralNetwork::ActivateHidden(MatrixXd& Mat)
{
    switch(HiddenActFun)
    {
        case linear:
            Linear(Mat);
            return;
            
        case sigmoid:
            Sigmoid(Mat);
            return;
            
        case relu:
            ReLU(Mat);
            return;
        case leakyrelu:
            LeakyReLU(Mat);
            return;
    }
    
    return;
}

void NeuralNetwork::ActivateOutput(MatrixXd& Mat)
{
    switch(OutputActFun)
    {
        case linear:
        {
            Linear(Mat);
            return;
        }
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
    }
    
    return;
}

void NeuralNetwork::Linear(MatrixXd& Mat)
{
    return;
}

void NeuralNetwork::Sigmoid(MatrixXd& Mat)
{
    MatrixXd Numerator = MatrixXd::Ones(Mat.rows(), Mat.cols());
    MatrixXd temp = -Mat;
    temp = temp.array().exp();
    MatrixXd Denominator = (MatrixXd::Ones(Mat.rows(), Mat.cols()) + temp);
    
    Mat = Numerator.array() / Denominator.array();
    
    return;
}

void NeuralNetwork::ReLU(MatrixXd& Mat)
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

void NeuralNetwork::LeakyReLU(MatrixXd& Mat)
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

vector<double> NeuralNetwork::Predict(const vector<vector<double>>& XTest)
{
    
    MatrixXd X = Utilities::ConvertToEigen(XTest);
    
    // Check for weights
    if(w1.isZero() || w2.isZero())
    {
        cout << endl << "Error in Predict() - No weights have been fit" << endl;
        return vector<double> (X.rows(),0);
        
    }
    
    // Add a column of ones
    X.conservativeResize(X.rows(), X.cols()+1);
    X.col(X.cols()-1) = VectorXd::Ones(X.rows());
    
    // Forward propagation
    MatrixXd Outputs = ForwardPropagation(X);
    
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


MatrixXd NeuralNetwork::ForwardPropagation(const MatrixXd& X)
{
    MatrixXd a2 = X * w1.transpose();
    ActivateHidden(a2);
    
    // Add a column of ones
    a2.conservativeResize(a2.rows(), a2.cols()+1);
    a2.col(a2.cols()-1) = VectorXd::Ones(a2.rows());
    
    MatrixXd a3 = a2 * w2.transpose();
    
    
    // Handle the activation function of the output units
    switch(CostFun)
    {
        case CrossEntropy:
        {
            ActivateOutput(a3);
            break;
        }
        case SumOfSquaredErrors:
        {
            break;
        }
    }
    
    return a3;
}


vector<double> NeuralNetwork::WinningOutput(const MatrixXd& Outputs)
{
    vector<double> Predictions(Outputs.rows(),0);
    
    for(int i=0; i < Outputs.rows(); ++i)
    {
        Outputs.row(i).maxCoeff( &Predictions[i] );
    }
    
    return Predictions;
}


void NeuralNetwork::CalculateCosts(const MatrixXd& Outputs, const MatrixXd& YTrain, int iter)
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


void NeuralNetwork::CrossEntropyCosts(const MatrixXd& Outputs, const MatrixXd& YTrain, const int& iter)
{
    MatrixXd LogOutputs = Outputs.array().log();
    MatrixXd term1 = -(YTrain.cwiseProduct(LogOutputs));
    
    MatrixXd logOneMinusOutputs = (MatrixXd::Ones(Outputs.rows(), Outputs.cols()) - Outputs).array().log();
    MatrixXd term2 = (MatrixXd::Ones(YTrain.rows(),YTrain.cols()) - YTrain).cwiseProduct(logOneMinusOutputs);
    
    Costs[iter] = (term1 - term2).sum();
    Costs[iter] *= (1/numTrainExamples);
    
    return;
}


void NeuralNetwork::SumOfSquaredErrorsCosts(const MatrixXd& Outputs, const MatrixXd& YTrain, const int& iter)
{
    MatrixXd O = Outputs;
    MatrixXd Y = YTrain;
    Costs[iter] = ((O - Y).transpose()*(O - YTrain)).value();
    Costs[iter] *= (1/numTrainExamples);
    
    return;
}


void NeuralNetwork::Regularize(const int& iter)
{
    double RegTerm = 0;
    RegTerm += ((w1.block(0, 0, w1.rows(), w1.cols()-1)).cwiseProduct((w1.block(0, 0, w1.rows(), w1.cols()-1)))).sum();
    RegTerm += ((w2.block(0, 0, w2.rows(), w2.cols()-1)).cwiseProduct((w2.block(0, 0, w2.rows(), w2.cols()-1)))).sum();
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
    z2.col(z2.cols()-1) = VectorXd::Ones(z2.rows());
    
    // Get the layer 2 errors
    MatrixXd delta2 = delta3 * w2;
    ActivateHidden(z2);
    
    delta2 = (GetHiddenActivationGradient(z2)).cwiseProduct(delta2);
    delta2.transposeInPlace();
    delta2.conservativeResize(delta2.rows()-1, delta2.cols());
    
    // Calculate the two gradients
    MatrixXd grad1 = delta2 * XTrain;
    MatrixXd grad2 = delta3.transpose() * z2;
    
    grad1 *= 1/numTrainExamples;
    grad2 *= 1/numTrainExamples;
    
    // Regularise
    grad1.block(0, 0, grad1.rows(), grad1.cols()-1) += (w1.block(0, 0, w1.rows(), w1.cols()-1)) * (Lambda/numTrainExamples);
    
    grad2.block(0, 0, grad2.rows(), grad2.cols()-1) += (w2.block(0, 0, w2.rows(), w2.cols()-1)) * (Lambda/numTrainExamples);
    
    return make_pair(grad1, grad2);
}

vector<double> NeuralNetwork::GetCosts() const
{
    // Check for weights
    if(Costs.empty())
    {
        cout << endl << "Error in GetCosts() - No costs have been calculated" << endl;
    }
    
    return Costs;
}


MatrixXd NeuralNetwork::GetHiddenActivationGradient(const MatrixXd& Activations)
{
    switch(HiddenActFun)
    {
        case sigmoid:
        {
            return (MatrixXd::Ones(Activations.rows(), Activations.cols()) - Activations).cwiseProduct(Activations);
        }
        case relu:
        {
            MatrixXd Gradients = MatrixXd::Zero(Activations.rows(), Activations.cols());
            for(int c=0; c < Activations.cols(); ++c)
            {
                for(int r=0; r < Activations.rows(); ++r)
                {
                    if(Activations(r,c) > 0)
                    {
                        Gradients(r,c) = 1;
                    }
                }
            }
            return Gradients;
        }
        case leakyrelu:
        {
            MatrixXd Gradients = MatrixXd::Zero(Activations.rows(), Activations.cols());
            Gradients = Gradients.array() + 0.5;
            for(int c=0; c < Activations.cols(); ++c)
            {
                for(int r=0; r < Activations.rows(); ++r)
                {
                    if(Activations(r,c) > 0)
                    {
                        Gradients(r,c) = 1;
                    }
                }
            }
            return Gradients;
        }
        case linear:
        {
            return MatrixXd::Ones(Activations.rows(), Activations.cols());
        }
    }
}
