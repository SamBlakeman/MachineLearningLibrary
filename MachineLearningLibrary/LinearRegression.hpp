//
//  LinearRegression.hpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 05/05/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#ifndef LinearRegression_hpp
#define LinearRegression_hpp

#include <stdio.h>
#include <vector>
#include "SupervisedModel.hpp"

using namespace std;

enum OptimizationMethod {BatchGradientDescent, NormalEquations};

class LinearRegression : public SupervisedModel
{
public:
    
    // Constructors
    LinearRegression(double lambda, OptimizationMethod Op);
    LinearRegression(double lambda, double alpha, int iter, OptimizationMethod Op);
    LinearRegression(vector<double> weights);
    
    // Fit the weights of the model
    virtual void Fit(const vector<vector<double>>& XT, const vector<double>& YTrain) override;
    
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
    void NormalEquation(const vector<vector<double>>& XTrain, const vector<double>& YTrain);
    
    double Lambda = 0.f;
    double Alpha = 0.1;
    int Iterations = 0;
    double numFeatures = 0;
    double numTrainExamples = 0;
    vector<double> Weights;
    vector<double> Costs;
    OptimizationMethod Opt;
    
};

#endif /* LinearRegression_hpp */