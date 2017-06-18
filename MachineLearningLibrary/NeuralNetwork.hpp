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

#endif /* NeuralNetwork_hpp */

using namespace std;

class NeuralNetwork
{
public:
    
    // Constructors
    NeuralNetwork(double alpha, double lambda, int numHidden, int Iters);
    
    // Fit the weights of the model
    void Fit(vector<vector<double>> XTrain, const vector<vector<double>>& YTrain);
    void InitialiseWeights();
    
    //void GradientDescent(const vector<vector<double>>& XTrain, const vector<double>& YTrain);
    //void Sigmoid(vector<double>& Vec);
    
    // Predict
    vector<double> Predict(vector<vector<double>> XTest);
    //void Quantise(vector<double>& Probabilities);
    
    // Getters
    //vector<double> GetWeights();
    //vector<double> GetCosts();
    //double GetAccuracy(const vector<vector<double>>& X, const vector<double>& Y);
    
private:
    
    // Fit
    void Sigmoid(vector<double>& Vec);
    void Sigmoid(vector<vector<double>>& Mat);
    vector<vector<double>> ForwardPropagation(const vector<vector<double>>& X);
    void AddBiasUnit(vector<vector<double>>& Activations);
    void CalculateCosts(const vector<vector<double>>& Outputs, const vector<vector<double>>& YTrain, int iter);
    pair<vector<vector<double>>,vector<vector<double>>> CalculateGradients(const vector<vector<double>>& Outputs, const vector<vector<double>>& XTrain, const vector<vector<double>>& YTrain);
    
    
    // Predict
    vector<double> WinningOutput(vector<vector<double>> Outputs);
    
    double Alpha = 0.1;
    double Lambda = 0.f;
    int numOut = 0;
    int numHid = 0;
    double numFeatures = 0;
    double numTrainExamples = 0;
    int Iterations = 0;
    vector<vector<double>> w1;
    vector<vector<double>> w2;
    vector<double> Costs;
    
    
    
};