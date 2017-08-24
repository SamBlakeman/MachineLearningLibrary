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

using namespace std;

enum OptimizationMethod {BatchGradientDescent, NormalEquations};

class LinearRegression
{
public:
    
    // Constructors
    LinearRegression(double lambda, OptimizationMethod Op);
    LinearRegression(double lambda, double alpha, int iter, OptimizationMethod Op);
    LinearRegression(vector<double> weights);
    
    // Fit the weights of the model
    void Fit(vector<vector<double>> XTrain, const vector<double>& YTrain);
    
    // Predict
    vector<double> Predict(vector<vector<double>> XTest);
    
    // Goodness of fit
    double CalculateRSquared(vector<vector<double>> X, const vector<double>& Y);
    
    // Getters
    vector<double> GetWeights();
    vector<double> GetCosts();
    
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