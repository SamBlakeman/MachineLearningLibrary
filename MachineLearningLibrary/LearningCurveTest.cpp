//
//  LearningCurveTest.cpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 30/09/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#include "LearningCurveTest.hpp"
#include "LogisticRegression.hpp"
#include "PreProcessing.hpp"
#include "Utilities.hpp"
#include <iostream>

void LearningCurveTest::Run()
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
    int numFolds = 3;
    LearningCurveResults Results = lgr.LearningCurve(X, Y, 5, numFolds);
    
    // Save the results for plotting
    string name = "/Users/samblakeman/Desktop/LearningCurveTrainMeanAccuracies.txt";
    auto filename = name.c_str();
    Utilities::SaveVectorAsCSV(Results.TrainMeanPerformance, filename);
    
    name = "/Users/samblakeman/Desktop/LearningCurveTrainStdAccuracies.txt";
    filename = name.c_str();
    Utilities::SaveVectorAsCSV(Results.TrainStdPerformance, filename);
    
    name = "/Users/samblakeman/Desktop/LearningCurveTestMeanAccuracies.txt";
    filename = name.c_str();
    Utilities::SaveVectorAsCSV(Results.TestMeanPerformance, filename);
    
    name = "/Users/samblakeman/Desktop/LearningCurveTestStdAccuracies.txt";
    filename = name.c_str();
    Utilities::SaveVectorAsCSV(Results.TestStdPerformance, filename);
    
    return;
}