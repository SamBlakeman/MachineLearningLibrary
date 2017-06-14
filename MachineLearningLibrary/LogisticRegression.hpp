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

#endif /* LogisticRegression_hpp */

using namespace std;



class LogisticRegression
{
public:
    
    // Constructors
    LogisticRegression(double lambda, double alpha, int iter);
    LogisticRegression(vector<double> weights, double lambda, double alpha, int iter);
    
    // Fit the weights of the model
    void Fit(vector<vector<double>> XTrain, const vector<double>& YTrain);
    
    // Predict
    vector<double> Predict(vector<vector<double>> XTest);
    
    // Getters
    vector<double> GetWeights();
    vector<double> GetCosts();
    double GetAccuracy(const vector<vector<double>>& X, const vector<double>& Y);

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
