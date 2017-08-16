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

enum ActivationFunction {linear, sigmoid, relu};

using namespace std;
using namespace Eigen;

class NeuralNetwork
{
public:
    
    // Constructors
    NeuralNetwork(double alpha, double lambda, int numHidden, int numOutput, int Iters);
    NeuralNetwork(double alpha, double lambda, int numHidden, int numOutput, int Iters, ActivationFunction HiddenActivation);
    
    // Fit the weights of the model
    void Fit(const vector<vector<double>>& X, const vector<vector<double>>& Y);
    void InitialiseWeights();
    
    // Predict
    vector<int> Predict(const vector<vector<double>>& XTest);
    
    // Getters
    //vector<double> GetWeights();
    vector<double> GetCosts() const;
    //double GetAccuracy(const vector<vector<double>>& X, const vector<double>& Y);
    
private:
    
    // Fit
    pair<MatrixXd,MatrixXd> ConvertToEigen(const vector<vector<double>>& XTrain, const vector<vector<double>>& YTrain );
    MatrixXd ConvertToEigen(const vector<vector<double>>& X);
    void Linear(MatrixXd& Mat);
    void Sigmoid(MatrixXd& Mat);
    void ReLU(MatrixXd& Mat);
    MatrixXd ForwardPropagation(const MatrixXd& X);
    void CalculateCosts(const MatrixXd& Outputs, const MatrixXd& YTrain, int iter);
    pair<MatrixXd,MatrixXd> CalculateGradients(const MatrixXd& Outputs, const MatrixXd& XTrain, const MatrixXd& YTrain);
    void ActivateHidden(MatrixXd& Mat);
    void ActivateOutput(MatrixXd& Mat);
    MatrixXd GetHiddenActivationGradient(const MatrixXd& Activations);
    
    
    // Predict
    vector<int> WinningOutput(const MatrixXd& Outputs);
    
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