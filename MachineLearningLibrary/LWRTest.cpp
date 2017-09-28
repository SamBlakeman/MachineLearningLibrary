//
//  LWRTest.cpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 23/08/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#include "LWRTest.hpp"
#include "Utilities.hpp"
#include "PreProcessing.hpp"
#include "LocallyWeightedRegression.hpp"
#include <vector>
#include <iostream>
#include <string>

using namespace std;

void LWRTest::Run()
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
    
    // Fit locally weighted regression model
    LocallyWeightedRegression lwr(.1);
    lwr.Fit(XTrain, YTrain);
    
    // Calculate R squared
    double RSq = lwr.CalculatePerformance(XTest, YTest);
    cout << endl << "R Squared:\n" << RSq << endl;
    
    // Save the predictions and the actual values
    vector<double> Predictions = lwr.Predict(XTest);
    cout << "Saving predictions\n";
    string name = "/Users/samblakeman/Desktop/Predictions.txt";
    auto filename = name.c_str();
    Utilities::SaveVectorAsCSV(Predictions, filename);
    
    cout << "Saving test values\n";
    name = "/Users/samblakeman/Desktop/YTest.txt";
    filename = name.c_str();
    Utilities::SaveVectorAsCSV(YTest, filename);

    
    return;
    
    
    
}