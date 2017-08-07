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

    // Train Network
    double alpha = 0.001;
    double lambda = 0;
    int numHidden = 50;
    int numOutput = 10;
    int Iters = 100;
    ActivationFunction AF = sigmoid;
    
    // One hot encode Y
    vector<vector<double>> YTrainEnc = pp.OneHotEncoding(YTrain, numOutput);
    
    NeuralNetwork nn(alpha, lambda, numHidden, numOutput, Iters, AF);
    nn.Fit(XTrain, YTrainEnc);
    
    cout << "Saving costs\n";
    
    // Save the costs for plotting
    vector<double> Costs = nn.GetCosts();
    string name = "/Users/samblakeman/Desktop/NNCosts.txt";
    auto filename = name.c_str();
    Utilities::SaveVectorAsCSV(Costs, filename);
    
    
    
    
    return;
}