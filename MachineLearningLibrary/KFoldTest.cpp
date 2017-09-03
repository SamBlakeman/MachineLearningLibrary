//
//  KFoldTest.cpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 03/09/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#include "KFoldTest.hpp"
#include "LogisticRegression.hpp"
#include "PreProcessing.hpp"
#include "Utilities.hpp"
#include <iostream>


void KFoldTest::Run()
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
    KFoldResults Results = lgr.KFoldCrossValidation(X, Y, numFolds);
    
    // Print KFold Results
    cout << "\nTraining RSquared:\n" << Results.TrainingRSquared << endl;
    cout << "\nTraining Accuracy:\n" << Results.TrainingAccuracy << endl;
    cout << "\nTest RSquared:\n" << Results.TestRSquared << endl;
    cout << "\nTest Accuracy:\n" << Results.TestAccuracy << endl;
    
    return;
    
    
}