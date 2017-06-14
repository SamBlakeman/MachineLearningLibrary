//
//  LogisticRegression.cpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 20/05/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#include "LogisticRegression.hpp"
#include "Utilities.hpp"
#include <cmath>
#include <numeric>
#include <iostream>


LogisticRegression::LogisticRegression(double lambda, double alpha, int iter)
{
    Lambda = lambda;
    Alpha = alpha;
    Iterations = iter;
    return;
}

LogisticRegression::LogisticRegression(vector<double> weights, double lambda, double alpha, int iter)
{
    Weights = weights;
    Alpha = alpha;
    Iterations = iter;
    return;
}


void LogisticRegression::Fit(vector<vector<double>> XTrain, const vector<double>& YTrain)
{
    numFeatures = XTrain[0].size() + 1;
    numTrainExamples = XTrain.size();
    
    // Add a column of ones
    for(int i = 0; i < XTrain.size(); ++i)
    {
        XTrain[i].push_back(1);
    }
    
    // Zero initialise the weights
    vector<double> W_init(numFeatures, 0);
    Weights = W_init;
    
    // Zero initialise the costs
    vector<double> C_init(Iterations, 0);
    Costs = C_init;
    
    // Run gradient descent
    GradientDescent(XTrain, YTrain);
    
    return;
}


void LogisticRegression::GradientDescent(const vector<vector<double>>& XTrain, const vector<double>& YTrain)
{
    vector<double> delta(numFeatures);
    vector<double> Hypotheses;
    double RegTerm;
    
    for(int iter=0; iter < Iterations; ++iter)
    {        
        // Calculate cost
        Hypotheses = Utilities::Product(XTrain, Weights);
        Sigmoid(Hypotheses);
        
        for(int i = 0; i < numTrainExamples; ++i)
        {
            Costs[iter] += (YTrain[i]*log(Hypotheses[i])) + ((1-YTrain[i])*log(1-Hypotheses[i]));
        }
        Costs[iter] *= -(1/numTrainExamples);
        
        RegTerm = 0;
        for(int i = 0; i < numFeatures-1; ++i)
        {
            RegTerm += pow(Weights[i], 2);
        }
        RegTerm *= Lambda/(2*numTrainExamples);
        Costs[iter] += RegTerm;
        
        
        
        //delta = (1/m) * (X'*(X*theta - y));
        Hypotheses = Utilities::Product(XTrain, Weights);
        Sigmoid(Hypotheses);
        delta = Utilities::Product(Utilities::Transpose(XTrain),Utilities::VecSub(Hypotheses, YTrain));
        delta = Utilities::ScalarMult(delta, (1/numTrainExamples));
        
        //regularization term
        delta = Utilities::VecAdd(delta, Utilities::ScalarMult(Weights, Lambda/numTrainExamples));
        delta[delta.size()-1] = delta[delta.size()-1] - (Weights[delta.size()-1] * Lambda/numTrainExamples);
        
        //theta = theta - alpha * delta;
        Weights = Utilities::VecSub(Weights, Utilities::ScalarMult(delta, Alpha));

    }
    
    return;
}

void LogisticRegression::Sigmoid(vector<double>& Vec)
{
    for(int i = 0; i < Vec.size(); ++i)
    {
        Vec[i] = 1/(1+exp(-Vec[i]));
    }
    
    return;
}


vector<double> LogisticRegression::Predict(vector<vector<double>> XTest)
{
    vector<double> Predictions(XTest.size(),0);
    
    // Check for weights
    if(Weights.empty())
    {
        cout << endl << "Error in Predict() - No weights have been fit" << endl;
        return Predictions;
        
    }
    
    // Add a column of ones
    for(int i = 0; i < XTest.size(); ++i)
    {
        XTest[i].push_back(1);
    }
    
    Predictions = Utilities::Product(XTest, Weights);
    Sigmoid(Predictions);
    Quantise(Predictions);
    
    return Predictions;
}

void LogisticRegression::Quantise(vector<double>& Probabilities)
{
    for (int p = 0; p < Probabilities.size(); ++p)
    {
        Probabilities[p] = round(Probabilities[p]);
    }
}


vector<double> LogisticRegression::GetWeights()
{
    // Check for weights
    if(Weights.empty())
    {
        cout << endl << "Error in GetWeights() - No weights have been fit" << endl;
    }
    
    return Weights;
}


vector<double> LogisticRegression::GetCosts()
{
    // Check for weights
    if(Costs.empty())
    {
        cout << endl << "Error in GetCosts() - No costs have been calculated from gradient descent" << endl;
    }
    
    return Costs;
}

double LogisticRegression::GetAccuracy(const vector<vector<double>>& X, const vector<double>& Y)
{
    double Accuracy;
    
    vector<double> Predictions = Predict(X);
    
    double numCorrect = 0;
    double total = Y.size();
    
    for(int e = 0; e < total; ++e)
    {
        if(Predictions[e] == Y[e])
        {
            ++numCorrect;
        }
    }
    
    Accuracy = (numCorrect/total)*100;
    
    return Accuracy;
}