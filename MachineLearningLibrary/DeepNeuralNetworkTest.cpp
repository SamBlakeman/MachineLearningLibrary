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


void DeepNeuralNetworkTest::RunClassificationTest()
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
    CostFunction CostFun = CrossEntropy;
    ActivationFunction AF = sigmoid;
    
    DeepNeuralNetwork dnn(alpha, lambda, numOutput, Iters, CostFun);
    dnn.AddDenseLayer(50, int(XTrain[0].size()), AF);
    dnn.AddDenseLayer(30, 50, AF);
    dnn.AddDenseLayer(20, 30, AF);
    
    // Train Network
    dnn.Fit(XTrain, YTrain);
    
    // Print accuracy on test set
    double Accuracy = dnn.CalculatePerformance(XTest, YTest);
    cout << "\nTest Accuracy = " << Accuracy << endl;
    
    // Save the costs for plotting
    cout << "Saving costs";
    vector<double> Costs = dnn.GetCosts();
    string name = "/Users/samblakeman/Desktop/Costs.txt";
    auto filename = name.c_str();
    Utilities::SaveVectorAsCSV(Costs, filename);
    
    // Save the predictions and the actual values
    vector<double> Predictions = dnn.Predict(XTest);
    cout << "Saving predictions";
    name = "/Users/samblakeman/Desktop/Predictions.txt";
    filename = name.c_str();
    Utilities::SaveVectorAsCSV(Predictions, filename);
    
    cout << "Saving actual values";
    name = "/Users/samblakeman/Desktop/YTest.txt";
    filename = name.c_str();
    Utilities::SaveVectorAsCSV(YTest, filename);
    
    
    return;
    
}


void DeepNeuralNetworkTest::RunRegressionTest()
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
    double alpha = 0.00000001;
    double lambda = 0;
    int numOutput = 1;
    int Iters = 1000;
    CostFunction CostFun = SumOfSquaredErrors;
    ActivationFunction AF = leakyrelu;
    
    DeepNeuralNetwork dnn(alpha, lambda, numOutput, Iters, CostFun);
    dnn.AddDenseLayer(50, int(XTrain[0].size()), AF);
    dnn.AddDenseLayer(30, 50, AF);
    dnn.AddDenseLayer(20, 30, AF);
    
    // Train Network
    dnn.Fit(XTrain, YTrain);
    
    // Calculate R squared
    double RSq = dnn.CalculatePerformance(XTest, YTest);
    cout << endl << "R Squared:\n" << RSq << endl;
    
    // Save the costs for plotting
    cout << "Saving costs\n";
    vector<double> Costs = dnn.GetCosts();
    string name = "/Users/samblakeman/Desktop/Costs.txt";
    auto filename = name.c_str();
    Utilities::SaveVectorAsCSV(Costs, filename);
    
    // Save the predictions and the actual values
    cout << "Saving predictions\n";
    vector<double> Predictions = dnn.Predict(XTest);
    name = "/Users/samblakeman/Desktop/Predictions.txt";
    filename = name.c_str();
    Utilities::SaveVectorAsCSV(Predictions, filename);
    
    cout << "Saving actual values\n";
    name = "/Users/samblakeman/Desktop/YTest.txt";
    filename = name.c_str();
    Utilities::SaveVectorAsCSV(YTest, filename);
    
    
    
    return;
}