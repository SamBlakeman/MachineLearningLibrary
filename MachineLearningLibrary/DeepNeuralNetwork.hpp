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

enum ActivationFunction {linear, sigmoid, relu};

class DenseLayer
{
public:
    
    DenseLayer(int NumberOfUnits, int NumberOfInputs);
    DenseLayer(int NumberOfUnits, int NumberOfInputs, ActivationFunction Activation);
    void InitialiseWeights(int NumberOfOutputs);
    int GetNumberOfUnits() const;
    double GetPenalty() const;
    MatrixXd GetWeights() const;
    void Propagate(MatrixXd& Inputs);
    
private:
    
    void Activate(MatrixXd& Mat);
    void Linear(MatrixXd& Mat);
    void Sigmoid(MatrixXd& Mat);
    void ReLU(MatrixXd& Mat);
    
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
    
    DeepNeuralNetwork(double alpha, double lambda, int numOutput, int Iters);
    void AddDenseLayer(int NumberOfUnits, int NumberOfInputs);
    void Fit(const vector<vector<double>>& X, const vector<vector<double>>& Y);
    vector<double> GetCosts() const;
    
    
private:
    
    void InitialiseHiddenWeights();
    void InitialiseOutputWeights();
    MatrixXd ForwardPropagation(const MatrixXd& X);
    void Sigmoid(MatrixXd& Mat);
    void CalculateCosts(const MatrixXd& Outputs, const MatrixXd& YTrain, int iter);
    vector<MatrixXd> CalculateGradients(const MatrixXd& Outputs, const MatrixXd& XTrain, const MatrixXd& YTrain);
    pair<MatrixXd,MatrixXd> ConvertToEigen(const vector<vector<double>>& XTrain, const vector<vector<double>>& YTrain );
    
    double Alpha = 0.1;
    double Lambda = 0.f;
    int numOut = 0;
    double numFeatures = 0;
    double numTrainExamples = 0;
    int Iterations = 0;
    MatrixXd OutputWeights;
    vector<double> Costs;
    vector<DenseLayer> HiddenLayers;
    
};









#endif /* DeepNeuralNetwork_hpp */
