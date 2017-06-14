//
//  LRTest.cpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 06/05/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#include "LRTest.hpp"
#include "PreProcessing.hpp"
#include "Utilities.hpp"
#include "LinearRegression.hpp"
#include <vector>
#include <iostream>

using namespace std;

void LRTest::Run()
{
 
    vector<double> ExampleOne = {84, 46};
    vector<double> ExampleTwo = {70, 30};
    vector<double> ExampleThree = {76, 57};
    vector<double> ExampleFour = {69, 25};
    vector<double> ExampleFive = {63, 28};
    vector<double> ExampleSix = {72, 36};
    vector<double> ExampleSeven = {79, 57};
    vector<double> ExampleEight = {75, 44};
    vector<double> ExampleNine = {73, 20};
    vector<double> ExampleTen = {65, 52};
    
    vector<vector<double>> X = {ExampleOne,ExampleTwo,ExampleThree,ExampleFour,ExampleFive,ExampleSix,ExampleSeven,ExampleEight,ExampleNine,ExampleTen};
    
    vector<double> Y = {354, 263, 451, 302, 288, 385, 402, 365, 190, 405};
    
    PreProcessing PP;
    
    // Split
    auto Result = PP.GetTrainAndTest(X, Y, .8f);
    auto Xs = Result.first;
    auto Ys = Result.second;
    
    vector<vector<double>> XTrain = Xs[0];
    vector<vector<double>> XTest = Xs[1];
    
    vector<double> YTrain = Ys[0];
    vector<double> YTest = Ys[1];
    
    // Standardize
    /*PP.StandardiseFit(XTrain);
    PP.StandardiseTransform(XTrain);
    PP.StandardiseTransform(XTest);*/
    
    // Normalise
    PP.NormaliseFit(XTrain);
    PP.NormaliseTransform(XTrain);
    PP.NormaliseTransform(XTest);
    
    // Print results
    Utilities::Print2DVector(XTrain);
    cout << endl;
    Utilities::Print2DVector(XTest);
    
    
    // Fit linear regression model
    LinearRegression lr(0, .01, 10000);
    lr.Fit(XTrain, YTrain);
    
    vector<double> c = lr.GetCosts();
    cout << endl << "Weights:\n";
    Utilities::PrintVector(c);
    
    vector<double> w = lr.GetWeights();
    cout << endl << "Weights:\n";
    Utilities::PrintVector(w);
    
    vector<double> Predictions = lr.Predict(XTest);
    cout << endl << "Predictions:\n";
    Utilities::PrintVector(Predictions);
    
    cout << endl << "Actual:\n";
    Utilities::PrintVector(YTest);
    
    PP.NormaliseTransform(X);
    double RSq = lr.CalculateRSquared(X, Y);
    cout << endl << "R Squared:\n" << RSq << endl;
    
    return;
}