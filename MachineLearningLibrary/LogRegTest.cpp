//
//  LogRegTest.cpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 23/05/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#include "LogRegTest.hpp"
#include "Utilities.hpp"
#include "PreProcessing.hpp"
#include "LogisticRegression.hpp"
#include <string>
#include <iostream>

void LogRegTest::Test1()
{
    
    string input = "/Users/samblakeman/Desktop/ex2data1.txt";
    const char* InputFile = input.c_str();
    
    vector<vector<double>> FeatureVector;
    
    FeatureVector = Utilities::ReadCSVFeatureVector(InputFile);
    
    // Separate
    PreProcessing pp;
    auto Separated = pp.SeperateXandY(FeatureVector);
    vector<vector<double>> X = Separated.first;
    vector<double> Y = Separated.second;
    
    // Split
    auto Split = pp.GetTrainAndTest(X, Y, .8);
    auto Xs = Split.first;
    auto Ys = Split.second;
    
    vector<vector<double>> XTrain = Xs[0];
    vector<vector<double>> XTest = Xs[1];
    vector<double> YTrain = Ys[0];
    vector<double> YTest = Ys[1];

    // Standardise
    pp.NormaliseFit(XTrain);
    pp.NormaliseTransform(XTrain);
    pp.NormaliseTransform(XTest);
    
    // Fit Model
    LogisticRegression logr(1, .01, 10000);
    logr.Fit(XTrain, YTrain);
    
    // Accuracy
    double Accuracy = logr.GetAccuracy(XTrain, YTrain);
    cout << "Training Accuracy:\n" << Accuracy << endl;
    
    Accuracy = logr.GetAccuracy(XTest, YTest);
    cout << "Test Accuracy:\n" << Accuracy << endl;
    
    vector<double> w = logr.GetWeights();
    cout << "Weights:" << endl;
    Utilities::PrintVector(w);
    
    vector<double> c = logr.GetCosts();
    
    string output = "/Users/samblakeman/Desktop/Costs.txt";
    const char* OutputFile = output.c_str();
    Utilities::SaveVectorAsCSV(c, OutputFile);
    
}

void LogRegTest::Test2()
{
    string input = "/Users/samblakeman/Desktop/pimadiabetes.txt";
    const char* InputFile = input.c_str();
    
    vector<vector<double>> FeatureVector;
    
    FeatureVector = Utilities::ReadCSVFeatureVector(InputFile);
    
    // Separate
    PreProcessing pp;
    auto Separated = pp.SeperateXandY(FeatureVector);
    vector<vector<double>> X = Separated.first;
    vector<double> Y = Separated.second;
    
    // Split
    auto Split = pp.GetTrainAndTest(X, Y, .85);
    auto Xs = Split.first;
    auto Ys = Split.second;
    
    vector<vector<double>> XTrain = Xs[0];
    vector<vector<double>> XTest = Xs[1];
    vector<double> YTrain = Ys[0];
    vector<double> YTest = Ys[1];
    
    
    // Normalise
    pp.NormaliseFit(XTrain);
    pp.NormaliseTransform(XTrain);
    pp.NormaliseTransform(XTest);
    
    // Train
    LogisticRegression lgr(1, .1, 10000);
    lgr.Fit(XTrain, YTrain);
    
    double Accuracy = lgr.GetAccuracy(XTest, YTest);
    cout << "\nAccuracy:\n" << Accuracy << endl;
    
    vector<double> Costs = lgr.GetCosts();
    string name = "/Users/samblakeman/Desktop/Costs.txt";
    auto filename = name.c_str();
    Utilities::SaveVectorAsCSV(Costs, filename);
    
    return;
}