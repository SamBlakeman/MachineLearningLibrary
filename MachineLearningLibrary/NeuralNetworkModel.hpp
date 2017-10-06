//
//  NeuralNetworkModel.hpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 06/10/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#ifndef NeuralNetworkModel_hpp
#define NeuralNetworkModel_hpp

#include <stdio.h>
#include <vector>
#include "Eigen/Dense"

using namespace std;
using namespace Eigen;

enum ActivationFunction {linear, sigmoid, relu, leakyrelu};


class DenseLayer
{
public:
    
    DenseLayer(int NumberOfUnits, int NumberOfInputs);
    DenseLayer(int NumberOfUnits, int NumberOfInputs, ActivationFunction Activation);
    void InitialiseWeights(int NumberOfOutputs);
    int GetNumberOfUnits() const;
    double GetPenalty() const;
    MatrixXd GetWeights() const;
    MatrixXd GetActivations() const;
    MatrixXd GetHiddenActivationGradient();
    void Propagate(MatrixXd& Inputs);
    void UpdateWeights(const MatrixXd& Grad, double Alpha);
    
private:
    
    void Activate(MatrixXd& Mat);
    void Linear(MatrixXd& Mat);
    void Sigmoid(MatrixXd& Mat);
    void ReLU(MatrixXd& Mat);
    void LeakyReLU(MatrixXd& Mat);
    
    MatrixXd w;
    double numInputs;
    double numUnits;
    double numOutputs;
    ActivationFunction ActFun = sigmoid;
    MatrixXd A;
    
};


class NeuralNetworkModel
{
public:
    
    // Addition of layers
    void AddDenseLayer(int NumberOfUnits, int NumberOfInputs);
    void AddDenseLayer(int NumberOfUnits, int NumberOfInputs, ActivationFunction ActFun);
    
    // Getters
    vector<double> GetCosts() const;
    
protected:
    
    void Sigmoid(MatrixXd& Mat);
    
    void InitialiseHiddenWeights();
    void InitialiseOutputWeights();
    MatrixXd ForwardPropagation(const MatrixXd& X);
    
    void CrossEntropyCosts(const MatrixXd& Outputs, const MatrixXd& YTrain, const int& iter);
    void SumOfSquaredErrorsCosts(const MatrixXd& Outputs, const MatrixXd& YTrain, const int& iter);
    
    void Regularize(const int& iter);
    vector<MatrixXd> CalculateGradients(const MatrixXd& Outputs, const MatrixXd& XTrain, const MatrixXd& YTrain);
    void UpdateLayers(const vector<MatrixXd>& Grads);
    
    double Alpha = 0.1;
    double Lambda = 0.f;
    int numOut = 0;
    double numFeatures = 0;
    double numTrainExamples = 0;
    int Iterations = 0;
    MatrixXd OutputWeights;
    vector<double> Costs;
    vector<DenseLayer> HiddenLayers;
    
private:
    
    
    
};

#endif /* NeuralNetworkModel_hpp */
