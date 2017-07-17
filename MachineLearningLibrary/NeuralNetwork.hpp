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

using namespace std;
using namespace Eigen;

enum BiasLocation{Row,Column};

class NeuralNetwork
{
public:
    
    // Constructors
    NeuralNetwork(double alpha, double lambda, int numHidden, int numOutput, int Iters);
    
    // Fit the weights of the model
    void Fit(MatrixXd XTrain, const MatrixXd& YTrain);
    void InitialiseWeights();
    
    // Predict
    VectorXd Predict(MatrixXd XTest);
    
    // Getters
    //vector<double> GetWeights();
    vector<double> GetCosts();
    //double GetAccuracy(const vector<vector<double>>& X, const vector<double>& Y);
    
private:
    
    // Fit
    void Sigmoid(vector<double>& Vec);
    void Sigmoid(MatrixXd& Mat);
    MatrixXd ForwardPropagation(const MatrixXd& X);
    //void AddBiasUnit(vector<vector<double>>& Activations, BiasLocation Location);
    void CalculateCosts(const MatrixXd& Outputs, const MatrixXd& YTrain, int iter);
    pair<MatrixXd,MatrixXd> CalculateGradients(const MatrixXd& Outputs, const MatrixXd& XTrain, const MatrixXd& YTrain);
    
    
    // Predict
    VectorXd WinningOutput(MatrixXd Outputs);
    
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
    
    
    
};

#endif /* NeuralNetwork_hpp */