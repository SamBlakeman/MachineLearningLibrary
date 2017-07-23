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
    Costs = vector<double>(Iterations,0);
}



void NeuralNetwork::Fit(MatrixXd XTrain, const MatrixXd& YTrain)
{
    
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


void NeuralNetwork::InitialiseWeights()
{
    // Zero initialisation
    w1 = MatrixXd::Random(numHid, numFeatures+1);
    w2 = MatrixXd::Random(numOut, numHid+1);

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
    MatrixXd Numerator = MatrixXd::Ones(Mat.rows(), Mat.cols());
    MatrixXd temp = -Mat;
    temp = temp.array().exp();
    MatrixXd Denominator = (MatrixXd::Ones(Mat.rows(), Mat.cols()) + temp);
    
    Mat = Numerator.array() / Denominator.array();
    
    return;
}




VectorXd NeuralNetwork::Predict(MatrixXd XTest)
{
    // Check for weights
    if(w1.isZero() || w2.isZero())
    {
        cout << endl << "Error in Predict() - No weights have been fit" << endl;
        return VectorXd (XTest.rows(),0);
        
    }
    
    // Add a column of ones
    XTest.conservativeResize(XTest.rows(), XTest.cols()+1);
    XTest.col(XTest.cols()-1) = VectorXd::Ones(XTest.rows());
    
    // Forward propagation
    MatrixXd Outputs = ForwardPropagation(XTest);
    
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
    a2.col(a2.cols()-1) = VectorXd::Ones(a2.rows());
    
    MatrixXd a3 = a2 * w2.transpose();
    Sigmoid(a3);
    
    return a3;
}


VectorXd NeuralNetwork::WinningOutput(MatrixXd Outputs)
{
    VectorXd Predictions = VectorXd::Zero(Outputs.size());
    
    for(int i=0; i < Outputs.size(); ++i)
    {
        Outputs.row(i).maxCoeff( &Predictions(i) );
    }
    
    return Predictions;
}


void NeuralNetwork::CalculateCosts(const MatrixXd& Outputs, const MatrixXd& YTrain, int iter)
{
    MatrixXd LogOutputs = Outputs.array().log();
    MatrixXd term1 = -(YTrain.cwiseProduct(LogOutputs));
    
    MatrixXd logOneMinusOutputs = (MatrixXd::Ones(Outputs.rows(), Outputs.cols()) - Outputs).array().log();
    MatrixXd term2 = (MatrixXd::Ones(YTrain.rows(),YTrain.cols()) - YTrain).cwiseProduct(logOneMinusOutputs);
    
    Costs[iter] = (term1 - term2).sum();
    Costs[iter] *= (1/numTrainExamples);
    
    double RegTerm = 0;
    RegTerm += (w1.cwiseProduct(w1)).sum();
    RegTerm += (w2.cwiseProduct(w2)).sum();
    RegTerm *= Lambda/(2*numTrainExamples);
    Costs[iter] += RegTerm;
    
    cout << "Cost for iter " << iter << " = " << Costs[iter] << endl;
    
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
    Sigmoid(z2);
    
    delta2 = ((MatrixXd::Ones(z2.rows(), z2.cols()) - z2).cwiseProduct(z2)).cwiseProduct(delta2);
    delta2.transposeInPlace();
    delta2.conservativeResize(delta2.rows()-1, delta2.cols());
    
    // Calculate the two gradients
    MatrixXd grad1 = delta2 * XTrain;
    MatrixXd grad2 = delta3.transpose() * z2;
    
    grad1 *= 1/numTrainExamples;
    grad2 *= 1/numTrainExamples;
    
    // Regularise
    grad1 += w1 * (Lambda/numTrainExamples);
    grad2 += w2 * (Lambda/numTrainExamples);
    
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
