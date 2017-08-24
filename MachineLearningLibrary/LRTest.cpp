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
 
    string fn = "/Users/samblakeman/Desktop/Concrete_Data.csv";
    const char* FileName = fn.c_str();
    
    vector<vector<double>> FeatureVector = Utilities::ReadCSVFeatureVector(FileName);
    
    // Separate
    PreProcessing pp;
    YLocation yloc = LastColumn;
    auto Separated = pp.SeperateXandY(FeatureVector, yloc);
    vector<vector<double>> X = Separated.first;
    vector<double> Y = Separated.second;
    
    // Split
    auto Seperated = pp.GetTrainAndTest(X, Y, .8);
    auto Xs = Seperated.first;
    auto Ys = Seperated.second;
    
    vector<vector<double>> XTrain = Xs[0];
    vector<vector<double>> XTest = Xs[1];
    vector<double> YTrain = Ys[0];
    vector<double> YTest = Ys[1];
    
    // Normalise
    pp.NormaliseFit(XTrain);
    pp.NormaliseTransform(XTrain);
    pp.NormaliseTransform(XTest);
    
    // Fit linear regression model
    //LinearRegression lr(0, .01, 10000, BatchGradientDescent);
    LinearRegression lr(0, NormalEquations);
    lr.Fit(XTrain, YTrain);
    
    // Calculate R squared
    double RSq = lr.CalculateRSquared(XTest, YTest);
    cout << endl << "R Squared:\n" << RSq << endl;
    
//    // Save the costs for plotting
//    cout << "Saving costs\n";
//    vector<double> Costs = lr.GetCosts();
//    string name = "/Users/samblakeman/Desktop/Costs.txt";
//    auto filename = name.c_str();
//    Utilities::SaveVectorAsCSV(Costs, filename);
    
    // Save the predictions and the actual values
    vector<double> Predictions = lr.Predict(XTest);
    cout << "\nSaving predictions";
    string name = "/Users/samblakeman/Desktop/Predictions.txt";
    auto filename = name.c_str();
    Utilities::SaveVectorAsCSV(Predictions, filename);
    
    cout << "\nSaving test values";
    name = "/Users/samblakeman/Desktop/YTest.txt";
    filename = name.c_str();
    Utilities::SaveVectorAsCSV(YTest, filename);
    
    return;
}