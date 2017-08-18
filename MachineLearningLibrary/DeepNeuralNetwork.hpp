//
//  DeepNeuralNetwork.hpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 28/07/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#ifndef DeepNeuralNetwork_hpp
#define DeepNeuralNetwork_hpp

#include <stdio.h>
#include <vector>
#include "Eigen/Dense"

using namespace std;
using namespace Eigen;

enum ActivationFunction {linear, sigmoid, relu, leakyrelu};
enum CostFunction {CrossEntropy, SumOfSquaredErrors};

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


class DeepNeuralNetwork
{
public:
    
    // Construction
    DeepNeuralNetwork(double alpha, double lambda, int numOutput, int Iters, CostFunction Cost);
    void AddDenseLayer(int NumberOfUnits, int NumberOfInputs);
    void AddDenseLayer(int NumberOfUnits, int NumberOfInputs, ActivationFunction ActFun);
    
    // Pre-Training
    void PreTrain(const vector<vector<double>>& X);
    
    // Fit
    void Fit(const vector<vector<double>>& X, const vector<double>& Y);
    
    // Predict
    vector<int> Predict(const vector<vector<double>>& XTest);
    
    // Getters
    vector<double> GetCosts() const;
    double GetAccuracy(const vector<vector<double>>& X, const vector<double>& Y);
    
    
private:
    
    vector<vector<double>> OneHotEncode(const vector<double>& Y);
    void InitialiseHiddenWeights();
    void InitialiseOutputWeights();
    MatrixXd ForwardPropagation(const MatrixXd& X);
    void Sigmoid(MatrixXd& Mat);
    void CalculateCosts(const MatrixXd& Outputs, const MatrixXd& YTrain, const int& iter);
    void CrossEntropyCosts(const MatrixXd& Outputs, const MatrixXd& YTrain, const int& iter);
    void SumOfSquaredErrorsCosts(const MatrixXd& Outputs, const MatrixXd& YTrain, const int& iter);
    void Regularize(const int& iter);
    vector<MatrixXd> CalculateGradients(const MatrixXd& Outputs, const MatrixXd& XTrain, const MatrixXd& YTrain);
    void UpdateLayers(const vector<MatrixXd>& Grads);
    pair<MatrixXd,MatrixXd> ConvertToEigen(const vector<vector<double>>& XTrain, const vector<vector<double>>& YTrain );
    MatrixXd ConvertToEigen(const vector<vector<double>>& X);
    vector<int> WinningOutput(const MatrixXd& Outputs);
    
    double Alpha = 0.1;
    double Lambda = 0.f;
    int numOut = 0;
    double numFeatures = 0;
    double numTrainExamples = 0;
    int Iterations = 0;
    MatrixXd OutputWeights;
    vector<double> Costs;
    vector<DenseLayer> HiddenLayers;
    CostFunction CostFun;
    
};









#endif /* DeepNeuralNetwork_hpp */
