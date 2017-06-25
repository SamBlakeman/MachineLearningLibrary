//
//  NBCTest.cpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 29/05/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#include "NBCTest.hpp"
#include "Utilities.hpp"
#include "PreProcessing.hpp"
#include "NaiveBayesClassifier.hpp"
#include <vector>
#include <iostream>
#include <string>
#include <cmath>


void NBCTest::DiscreteTest()
{
    vector<vector<double>> X;
    
    X.push_back(vector<double> {3, 3, 2, 1});
    X.push_back(vector<double> {3, 3, 2, 2});
    X.push_back(vector<double> {2, 3, 2, 1});
    X.push_back(vector<double> {1, 2, 2, 1});
    X.push_back(vector<double> {1, 1, 1, 1});
    X.push_back(vector<double> {1, 1, 1, 2});
    X.push_back(vector<double> {2, 1, 1, 2});
    X.push_back(vector<double> {3, 2, 2, 1});
    X.push_back(vector<double> {3, 1, 1, 1});
    X.push_back(vector<double> {1, 2, 1, 1});
    
    vector<double> Y = {0,0,1,1,1,0,1,0,1,1};
    
    vector<vector<double>> XTest;
    XTest.push_back(vector<double> {3,2,1,2});
    XTest.push_back(vector<double> {2,2,2,2});
    XTest.push_back(vector<double> {2,3,1,1});
    XTest.push_back(vector<double> {1,2,2,2});
    XTest.push_back(vector<double> {3,1,2,2});
    
    vector<double> YTest = {1,1,1,0,0};
    
    NaiveBayesClassifier nbc;
    
    nbc.Fit(X, Y);
    
    vector<double> Predictions;
    Predictions = nbc.Predict(XTest);
    
    cout << "Predictions:\n";
    Utilities::PrintVector(Predictions);
    
    cout << "Accuracy:\n";
    double Accuracy = nbc.GetAccuracy(XTest, YTest);
    cout << Accuracy << "%\n\n";
    
    
    return;
}

void NBCTest::ContinuousTest()
{
    string input = "/Users/samblakeman/Desktop/pimadiabetes.txt";
    const char* InputFile = input.c_str();
    
    vector<vector<double>> FeatureVector;
    
    FeatureVector = Utilities::ReadCSVFeatureVector(InputFile);
    
    // Separate
    PreProcessing pp;
    YLocation yloc = LastColumn;
    auto Separated = pp.SeperateXandY(FeatureVector, yloc);
    vector<vector<double>> X = Separated.first;
    vector<double> Y = Separated.second;
    
    // Split
    auto Split = pp.GetTrainAndTest(X, Y, .67);
    auto Xs = Split.first;
    auto Ys = Split.second;
    
    vector<vector<double>> XTrain = Xs[0];
    vector<vector<double>> XTest = Xs[1];
    vector<double> YTrain = Ys[0];
    vector<double> YTest = Ys[1];
    
    // Train
    NaiveBayesClassifier nbc(true);
    
    nbc.Fit(XTrain, YTrain);
    double Accuracy = nbc.GetAccuracy(XTest, YTest);
    cout << "\nAccuracy:\n" << Accuracy << endl;
    
    return;
}