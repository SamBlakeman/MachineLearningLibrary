//
//  DeepNeuralNetworkTest.cpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 07/08/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#include "DeepNeuralNetworkTest.hpp"
#include "Utilities.hpp"
#include "PreProcessing.hpp"
#include "DeepNeuralNetwork.hpp"
#include <string>
#include <iostream>


void DeepNeuralNetworkTest::Run()
{
    
    string fn = "/Users/samblakeman/Desktop/WisconsinDataSet.csv";
    const char* FileName = fn.c_str();
    
    vector<vector<double>> FeatureVector = Utilities::ReadCSVFeatureVector(FileName);
    
    //FeatureVector.resize(6000);
    
    // Separate
    PreProcessing pp;
    YLocation yloc = FirstColumn;
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
    
    // Construct Network
    double alpha = 0.001;
    double lambda = 0;
    int numOutput = 2;
    int Iters = 1000;
    ActivationFunction AF = sigmoid;
    
    // One hot encode Y
    vector<vector<double>> YTrainEnc = pp.OneHotEncoding(YTrain, numOutput);
    
    DeepNeuralNetwork dnn(alpha, lambda, numOutput, Iters);
    dnn.AddDenseLayer(50, int(XTrain[0].size()), AF);
    dnn.AddDenseLayer(30, 50, AF);
    dnn.AddDenseLayer(20, 30, AF);
    
    // Train Network
    dnn.Fit(XTrain, YTrainEnc);
    
    // Save the costs for plotting
    cout << "Saving costs\n";
    vector<double> Costs = dnn.GetCosts();
    string name = "/Users/samblakeman/Desktop/NNCosts.txt";
    auto filename = name.c_str();
    Utilities::SaveVectorAsCSV(Costs, filename);
    
    // Try some predictions
    vector<int> Predictions = dnn.Predict(XTest);
    
    // Save the predictions and the actual values
    cout << "Saving predictions\n";
    name = "/Users/samblakeman/Desktop/DNNPredictions.txt";
    filename = name.c_str();
    Utilities::SaveVectorAsCSV(Predictions, filename);
    
    cout << "Saving predictions\n";
    name = "/Users/samblakeman/Desktop/YTest.txt";
    filename = name.c_str();
    Utilities::SaveVectorAsCSV(YTest, filename);
    
    
    return;
    
}