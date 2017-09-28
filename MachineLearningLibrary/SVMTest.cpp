//
//  SVMTest.cpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 30/08/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#include "SVMTest.hpp"
#include "Utilities.hpp"
#include "PreProcessing.hpp"
#include "SupportVectorMachine.hpp"
#include <vector>
#include <string>
#include <iostream>



void SVMTest::Test1()
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
    double C = 10;
    double Alpha = 0.0001;
    int Iters = 1000;
    
    SupportVectorMachine svm(C, Alpha, Iters);
    
    double Variance = 1;
    
    svm.AddGaussianKernel(Variance);
    
    svm.Fit(XTrain, YTrain);
    
    double Accuracy = svm.CalculatePerformance(XTest, YTest);
    cout << "\nAccuracy:\n" << Accuracy << endl;
    
    vector<double> Costs = svm.GetCosts();
    string name = "/Users/samblakeman/Desktop/Costs.txt";
    auto filename = name.c_str();
    Utilities::SaveVectorAsCSV(Costs, filename);
    
    return;
}

void SVMTest::Test2()
{
    string fn = "/Users/samblakeman/Desktop/WisconsinDataSet.csv";
    const char* FileName = fn.c_str();
    
    vector<vector<double>> FeatureVector = Utilities::ReadCSVFeatureVector(FileName);
    
    // Separate
    PreProcessing pp;
    YLocation yloc = FirstColumn;
    auto Separated = pp.SeperateXandY(FeatureVector, yloc);
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
    double C = 1;
    double Alpha = 0.0001;
    int Iters = 1000;
    
    SupportVectorMachine svm(C, Alpha, Iters);
    
    double Variance = 1;
    
    svm.AddGaussianKernel(Variance);
    
    svm.Fit(XTrain, YTrain);
    
    double Accuracy = svm.CalculatePerformance(XTest, YTest);
    cout << "\nAccuracy:\n" << Accuracy << endl;
    
    vector<double> Costs = svm.GetCosts();
    string name = "/Users/samblakeman/Desktop/Costs.txt";
    auto filename = name.c_str();
    Utilities::SaveVectorAsCSV(Costs, filename);
    
    return;
}