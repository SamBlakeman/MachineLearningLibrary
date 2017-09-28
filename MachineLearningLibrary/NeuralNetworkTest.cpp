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

void NeuralNetworkTest::RunClassificationTest()
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
    CostFunction Cost = CrossEntropy;
    ActivationFunction AF = leakyrelu;
    
    NeuralNetwork nn(alpha, lambda, numHidden, numOutput, Iters, Cost, AF);
    nn.Fit(XTrain, YTrain);
    
    // Print accuracy on test set
    double Accuracy = nn.CalculatePerformance(XTest, YTest);
    cout << "\nTest Accuracy = " << Accuracy << endl << endl;
    
    // Save the costs for plotting
    cout << "Saving costs";
    vector<double> Costs = nn.GetCosts();
    string name = "/Users/samblakeman/Desktop/Costs.txt";
    auto filename = name.c_str();
    Utilities::SaveVectorAsCSV(Costs, filename);
    
    // Save the predictions and the actual values
    vector<double> Predictions = nn.Predict(XTest);
    cout << "Saving predictions";
    name = "/Users/samblakeman/Desktop/Predictions.txt";
    filename = name.c_str();
    Utilities::SaveVectorAsCSV(Predictions, filename);
    
    cout << "Saving test values";
    name = "/Users/samblakeman/Desktop/YTest.txt";
    filename = name.c_str();
    Utilities::SaveVectorAsCSV(YTest, filename);
    
    
    return;
}


void NeuralNetworkTest::RunRegressionTest()
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
    
    // Construct Network
    double alpha = 0.01;
    double lambda = 0;
    int numHidden = 50;
    int numOutput = 1;
    int Iters = 1000;
    CostFunction Cost = SumOfSquaredErrors;
    ActivationFunction AF = leakyrelu;
    
    NeuralNetwork nn(alpha, lambda, numHidden, numOutput, Iters, Cost, AF);
    
    // Train Network
    nn.Fit(XTrain, YTrain);
    
    // Calculate R squared
    double RSq = nn.CalculatePerformance(XTest, YTest);
    cout << endl << "\nR Squared:\n" << RSq << endl << endl;
    
    // Save the costs for plotting
    cout << "Saving costs";
    vector<double> Costs = nn.GetCosts();
    string name = "/Users/samblakeman/Desktop/Costs.txt";
    auto filename = name.c_str();
    Utilities::SaveVectorAsCSV(Costs, filename);
    
    // Save the predictions and the actual values
    vector<double> Predictions = nn.Predict(XTest);
    cout << "Saving predictions";
    name = "/Users/samblakeman/Desktop/Predictions.txt";
    filename = name.c_str();
    Utilities::SaveVectorAsCSV(Predictions, filename);
    
    cout << "Saving test values";
    name = "/Users/samblakeman/Desktop/YTest.txt";
    filename = name.c_str();
    Utilities::SaveVectorAsCSV(YTest, filename);
    
    return;
}