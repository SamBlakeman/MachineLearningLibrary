//
//  LinearRegression.cpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 05/05/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#include "LinearRegression.hpp"
#include "Utilities.hpp"
#include <cmath>
#include <numeric>
#include <iostream>

LinearRegression::LinearRegression(double lambda, OptimizationMethod Op)
{
    Lambda = lambda;
    Opt = Op;
    CostFun = SumOfSquaredErrors;
    
    if(Op == BatchGradientDescent)
    {
        cout << "\nUse Constructor with Gradient Descent Parameters!\n";
    }
    
    return;
}


LinearRegression::LinearRegression(double lambda, double alpha, int iter, OptimizationMethod Op)
{
    Lambda = lambda;
    Alpha = alpha;
    Iterations = iter;
    Opt = Op;
    CostFun = SumOfSquaredErrors;
    
    if(Op == NormalEquations)
    {
        cout << "\nAlpha and Iter will be unused for Normal Equations, use other constructor in the future\n";
    }
    
    return;
}


LinearRegression::LinearRegression(vector<double> weights)
{
    Weights = weights;
    CostFun = SumOfSquaredErrors;
    return;
}

void LinearRegression::Fit(const vector<vector<double>>& XT, const vector<double>& YTrain)
{
    vector<vector<double>> XTrain = XT;
    
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
    
    // Run optimization
    switch(Opt)
    {
        case BatchGradientDescent:
            GradientDescent(XTrain, YTrain);
            return;
        case NormalEquations:
            NormalEquation(XTrain, YTrain);
            return;
    }
}

void LinearRegression::GradientDescent(const vector<vector<double>>& XTrain, const vector<double>& YTrain)
{
    vector<double> delta(numFeatures);
    vector<double> Hypotheses;
    double RegTerm;
    
    for(int iter=0; iter < Iterations; ++iter)
    {
        // Calculate cost
        Hypotheses = Utilities::Product(XTrain, Weights);
        for(int i = 0; i < numTrainExamples; ++i)
        {
            Costs[iter] += pow(Hypotheses[i] - YTrain[i], 2);
        }
        
        RegTerm = 0;
        for(int i = 0; i < numFeatures-1; ++i)
        {
            RegTerm += pow(Weights[i], 2);
        }
        RegTerm *= Lambda;
        Costs[iter] += RegTerm;
        Costs[iter] *= 1/(2*numTrainExamples);
        
        if(iter%50 == 0)
        {
            cout << "Cost for iter " << iter << " = " << Costs[iter] << endl;
        }
        
        //delta = (1/m) * (X'*(X*theta - y));
        Hypotheses = Utilities::Product(XTrain, Weights);
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

void LinearRegression::NormalEquation(const vector<vector<double>>& XTrain, const vector<double>& YTrain)
{
    MatrixXd LambdaMat = MatrixXd::Identity(numFeatures, numFeatures);
    LambdaMat = LambdaMat.array() * Lambda;
    LambdaMat(0,0) = 0; // do not regularise intercept term
    
    MatrixXd X = Utilities::ConvertToEigen(XTrain);
    MatrixXd Y = Utilities::ConvertToEigen(YTrain);
    
    VectorXd Theta = (((X.transpose()*X)+LambdaMat).inverse())*X.transpose()*Y;
    
    Weights = Utilities::ConvertFromEigen(Theta);
    
    return;
}

vector<double> LinearRegression::Predict(const vector<vector<double>>& XT)
{
    vector<vector<double>> XTest = XT;
    
    vector<double> Predictions(XTest.size(),0);
    
    // Check for weights
    if(Weights.empty())
    {
        cout << endl << "\nError in Predict() - No weights have been fit\n" << endl;
        return Predictions;
        
    }
    
    // Add a column of ones
    for(int i = 0; i < XTest.size(); ++i)
    {
        XTest[i].push_back(1);
    }
    
    Predictions = Utilities::Product(XTest, Weights);
    
    return Predictions;
}


vector<double> LinearRegression::GetWeights()
{
    // Check for weights
    if(Weights.empty())
    {
        cout << endl << "\nError in GetWeights() - No weights have been fit\n" << endl;
    }
    
    return Weights;
}


vector<double> LinearRegression::GetCosts()
{
    // Check for weights
    if(Costs.empty())
    {
        cout << endl << "\nError in GetCosts() - No costs have been calculated from optimization\n" << endl;
    }
    
    return Costs;
}

void LinearRegression::SetLambda(double lambda)
{
    Lambda = lambda;
}

void LinearRegression::SetAlpha(double alpha)
{
    Alpha = alpha;
}

void LinearRegression::SetIterations(int iters)
{
    Iterations = iters;
}

void LinearRegression::SetTau(double tau)
{
    
}

void LinearRegression::SetC(double c)
{
    
}

void LinearRegression::SetVar(double var)
{
    
}


