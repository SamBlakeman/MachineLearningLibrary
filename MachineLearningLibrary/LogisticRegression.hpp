//
//  LogisticRegression.hpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 20/05/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#ifndef LogisticRegression_hpp
#define LogisticRegression_hpp

#include <stdio.h>
#include <vector>
#include "MachineLearningModel.hpp"

using namespace std;

class LogisticRegression : public MachineLearningModel
{
public:
    
    // Constructors
    LogisticRegression(double lambda, double alpha, int iter);
    LogisticRegression(vector<double> weights);
    
    // Fit the weights of the model
    virtual void Fit(const vector<vector<double>>& XTrain, const vector<double>& YTrain) override;
    
    // Predict
    virtual vector<double> Predict(const vector<vector<double>>& XT) override;
    
    // Getters
    vector<double> GetWeights();
    vector<double> GetCosts();
    
    // Setters
    virtual void SetLambda(double lambda) override;
    virtual void SetAlpha(double alpha) override;
    virtual void SetIterations(int iters) override;
    virtual void SetTau(double tau) override;
    virtual void SetC(double c) override;
    virtual void SetVar(double var) override;

private:
    
    // Fit
    void GradientDescent(const vector<vector<double>>& XTrain, const vector<double>& YTrain);
    void Sigmoid(vector<double>& Vec);
    
    // Predict
    void Quantise(vector<double>& Probabilities);
    
    double Lambda = 0.f;
    double Alpha = 0.1;
    int Iterations = 0;
    double numFeatures = 0;
    double numTrainExamples = 0;
    vector<double> Weights;
    vector<double> Costs;
    
};

#endif /* LogisticRegression_hpp */
