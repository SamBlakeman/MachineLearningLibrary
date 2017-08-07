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
    
    string fn = "/Users/samblakeman/Desktop/mnist_train.csv";
    const char* FileName = fn.c_str();
    
    vector<vector<double>> FeatureVector = Utilities::ReadCSVFeatureVector(FileName);
    
    FeatureVector.resize(6000);
    
    // Separate
    PreProcessing pp;
    YLocation yloc = FirstColumn;
    auto Separated = pp.SeperateXandY(FeatureVector, yloc);
    vector<vector<double>> XTrain = Separated.first;
    vector<double> YTrain = Separated.second;
    
    // Normalise
    //pp.NormaliseFit(XTrain);
    //pp.NormaliseTransform(XTrain);
    
    // Construct Network
    double alpha = 0.001;
    double lambda = 0;
    int numOutput = 10;
    int Iters = 100;
    
    // One hot encode Y
    vector<vector<double>> YTrainEnc = pp.OneHotEncoding(YTrain, numOutput);
    
    DeepNeuralNetwork dnn(alpha, lambda, numOutput, Iters);
    dnn.AddDenseLayer(50, int(XTrain[0].size()));
    dnn.AddDenseLayer(50, 50);
    
    // Train Network
    dnn.Fit(XTrain, YTrainEnc);
    
    cout << "Saving costs\n";
    
    // Save the costs for plotting
    vector<double> Costs = dnn.GetCosts();
    string name = "/Users/samblakeman/Desktop/NNCosts.txt";
    auto filename = name.c_str();
    Utilities::SaveVectorAsCSV(Costs, filename);
    
    
    return;
    
}