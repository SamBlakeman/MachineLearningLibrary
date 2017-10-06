//
//  NeuralNetwork.hpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 06/06/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#ifndef NeuralNetwork_hpp
#define NeuralNetwork_hpp

#include <stdio.h>
#include <vector>
#include "Eigen/Dense"
#include "SupervisedModel.hpp"
#include "NeuralNetworkModel.hpp"

using namespace std;
using namespace Eigen;

class NeuralNetwork : public SupervisedModel
{
public:
    
    // Constructors
    NeuralNetwork(double alpha, double lambda, int numHidden, int numOutput, int Iters, CostFunction Cost);
    NeuralNetwork(double alpha, double lambda, int numHidden, int numOutput, int Iters, CostFunction Cost, ActivationFunction HiddenActivation);
    
    // Fit the weights of the model
    virtual void Fit(const vector<vector<double>>& X, const vector<double>& Y) override;
    void InitialiseWeights();
    
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
    
    // Fit
    vector<vector<double>> OneHotEncode(const vector<double>& Y);
    void Linear(MatrixXd& Mat);
    void Sigmoid(MatrixXd& Mat);
    void ReLU(MatrixXd& Mat);
    void LeakyReLU(MatrixXd& Mat);
    MatrixXd ForwardPropagation(const MatrixXd& X);
    void CalculateCosts(const MatrixXd& Outputs, const MatrixXd& YTrain, int iter);
    void CrossEntropyCosts(const MatrixXd& Outputs, const MatrixXd& YTrain, const int& iter);
    void SumOfSquaredErrorsCosts(const MatrixXd& Outputs, const MatrixXd& YTrain, const int& iter);
    void Regularize(const int& iter);
    pair<MatrixXd,MatrixXd> CalculateGradients(const MatrixXd& Outputs, const MatrixXd& XTrain, const MatrixXd& YTrain);
    void ActivateHidden(MatrixXd& Mat);
    void ActivateOutput(MatrixXd& Mat);
    MatrixXd GetHiddenActivationGradient(const MatrixXd& Activations);
    
    
    // Predict
    vector<double> WinningOutput(const MatrixXd& Outputs);
    
    double Alpha = 0.1;
    double Lambda = 0.f;
    int numOut = 0;
    int numHid = 0;
    double numFeatures = 0;
    double numTrainExamples = 0;
    int Iterations = 0;
    MatrixXd w1;
    MatrixXd w2;
    vector<double> Costs;
    ActivationFunction HiddenActFun = sigmoid;
    ActivationFunction OutputActFun = sigmoid;
    
};

#endif /* NeuralNetwork_hpp */