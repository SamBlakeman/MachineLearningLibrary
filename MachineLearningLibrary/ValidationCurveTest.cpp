//
//  ValidationCurveTest.cpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 27/09/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#include "ValidationCurveTest.hpp"
#include "LogisticRegression.hpp"
#include "PreProcessing.hpp"
#include "Utilities.hpp"
#include <iostream>

void ValidationCurveTest::Run()
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
    
    // Train Using KFold Cross Validation
    LogisticRegression lgr(1, .1, 10000);
    int numFolds = 10;
    ValidationCurveResults Results = lgr.ValidationCurve(X, Y, Alpha, vector<double> {0.001,.01,.1,1,10}, numFolds);

    // Save the results for plotting
    string name = "/Users/samblakeman/Desktop/ValidationCurveTrainMeanAccuracies.txt";
    auto filename = name.c_str();
    Utilities::SaveVectorAsCSV(Results.TrainMeanPerformance, filename);
    
    name = "/Users/samblakeman/Desktop/ValidationCurveTrainStdAccuracies.txt";
    filename = name.c_str();
    Utilities::SaveVectorAsCSV(Results.TrainStdPerformance, filename);
    
    name = "/Users/samblakeman/Desktop/ValidationCurveTestMeanAccuracies.txt";
    filename = name.c_str();
    Utilities::SaveVectorAsCSV(Results.TestMeanPerformance, filename);
    
    name = "/Users/samblakeman/Desktop/ValidationCurveTestStdAccuracies.txt";
    filename = name.c_str();
    Utilities::SaveVectorAsCSV(Results.TestStdPerformance, filename);
    
    return;
}