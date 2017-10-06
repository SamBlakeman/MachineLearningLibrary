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
#include "SupervisedModel.hpp"
#include "NeuralNetworkModel.hpp"

class DeepNeuralNetwork : public SupervisedModel, public NeuralNetworkModel
{
public:
    
    // Construction
    DeepNeuralNetwork(double alpha, double lambda, int numOutput, int Iters, CostFunction Cost);
    
    // Pre-Training
    void PreTrain(const vector<vector<double>>& X);
    
    // Fit
    virtual void Fit(const vector<vector<double>>& X, const vector<double>& Y) override;
    
    // Predict
    virtual vector<double> Predict(const vector<vector<double>>& XTest) override;
    
    // Setters
    virtual void SetLambda(double lambda) override;
    virtual void SetAlpha(double alpha) override;
    virtual void SetIterations(int iters) override;
    virtual void SetTau(double tau) override;
    virtual void SetC(double c) override;
    virtual void SetVar(double var) override;
    
    
private:
    
    vector<vector<double>> OneHotEncode(const vector<double>& Y);
    void CalculateCosts(const MatrixXd& Outputs, const MatrixXd& YTrain, const int& iter);
    vector<double> WinningOutput(const MatrixXd& Outputs);
    
};


#endif /* DeepNeuralNetwork_hpp */
