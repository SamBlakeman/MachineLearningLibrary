//
//  DeepAutoEncoderTest.cpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 23/10/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#include "DeepAutoEncoderTest.hpp"
#include "Utilities.hpp"
#include "PreProcessing.hpp"
#include "DeepAutoEncoder.hpp"
#include "LogisticRegression.hpp"
#include <iostream>

void DeepAutoEncoderTest::Run()
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
    
    
    
    // DeepAutoEncoder to Reduce Dimensionality
    double alpha = 0.01;
    double lambda = 0;
    double Iters = 1000;
    DeepAutoEncoder DeepAE(alpha, lambda, Iters);
    
    // Add Layers
    DeepAE.AddDenseLayer(6, int(XTrain[0].size()));
    DeepAE.AddDenseLayer(4, 6);
    DeepAE.AddDenseLayer(6, 4);
    
    // Fit
    DeepAE.Fit(XTrain);
    
    // Get Encodings
    vector<vector<double>> XTrainEnc = DeepAE.GetEncodedLayer(XTrain, 2);
    vector<vector<double>> XTestEnc = DeepAE.GetEncodedLayer(XTest, 2);
    
    // Train Logistic Regression Model
    LogisticRegression lgr(1, .1, 10000);
    lgr.Fit(XTrainEnc, YTrain);
    
    // Calculate Accuracy
    double Accuracy = lgr.CalculatePerformance(XTestEnc, YTest);
    cout << "\nAccuracy:\n" << Accuracy << endl;
    
    vector<double> Costs = lgr.GetCosts();
    string name = "/Users/samblakeman/Desktop/Costs.txt";
    auto filename = name.c_str();
    Utilities::SaveVectorAsCSV(Costs, filename);
    
    
    
    return;
}