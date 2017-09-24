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
#include "MachineLearningModel.hpp"

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


class DeepNeuralNetwork : public MachineLearningModel
{
public:
    
    // Construction
    DeepNeuralNetwork(double alpha, double lambda, int numOutput, int Iters, CostFunction Cost);
    void AddDenseLayer(int NumberOfUnits, int NumberOfInputs);
    void AddDenseLayer(int NumberOfUnits, int NumberOfInputs, ActivationFunction ActFun);
    
    // Pre-Training
    void PreTrain(const vector<vector<double>>& X);
    
    // Fit
    virtual void Fit(const vector<vector<double>>& X, const vector<double>& Y) override;
    
    // Predict
    virtual vector<double> Predict(const vector<vector<double>>& XTest) override;
    
    // Getters
    vector<double> GetCosts() const;
    
    // Setters
    virtual void SetLambda(double lambda) override;
    virtual void SetAlpha(double alpha) override;
    virtual void SetIterations(int iters) override;
    virtual void SetTau(double tau) override;
    virtual void SetC(double c) override;
    virtual void SetVar(double var) override;
    
    
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
    vector<double> WinningOutput(const MatrixXd& Outputs);
    
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
