//
//  NeuralNetworkTest.cpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 21/06/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#include "NeuralNetworkTest.hpp"
#include "Utilities.hpp"
#include "PreProcessing.hpp"
#include "NeuralNetwork.hpp"
#include <string>
#include <iostream>

void NeuralNetworkTest::Run()
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

    // Train Network
    double alpha = 0.01;
    double lambda = 0;
    int numHidden = 50;
    int numOutput = 2;
    int Iters = 1000;
    ActivationFunction AF = leakyrelu;
    
    NeuralNetwork nn(alpha, lambda, numHidden, numOutput, Iters, AF);
    nn.Fit(XTrain, YTrain);
    
    // Save the costs for plotting
    cout << "Saving costs\n";
    vector<double> Costs = nn.GetCosts();
    string name = "/Users/samblakeman/Desktop/NNCosts.txt";
    auto filename = name.c_str();
    Utilities::SaveVectorAsCSV(Costs, filename);
    
    // Print accuracy on test set
    double Accuracy = nn.GetAccuracy(XTest, YTest);
    cout << "Test Accuracy = " << Accuracy << endl;
    
    // Try some predictions
    vector<int> Predictions = nn.Predict(XTest);
    
    // Save the predictions and the actual values
    cout << "Saving predictions\n";
    name = "/Users/samblakeman/Desktop/DNNPredictions.txt";
    filename = name.c_str();
    Utilities::SaveVectorAsCSV(Predictions, filename);
    
    cout << "Saving test values\n";
    name = "/Users/samblakeman/Desktop/YTest.txt";
    filename = name.c_str();
    Utilities::SaveVectorAsCSV(YTest, filename);
    
    
    return;
}