//
//  TwoLayerNNTest.cpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 21/06/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#include "TwoLayerNNTest.hpp"
#include "Utilities.hpp"
#include "PreProcessing.hpp"
#include "NeuralNetwork.hpp"
#include <string>
#include <iostream>

void TwoLayerNNTest::Run()
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
    
    // Scale
    //pp.NormaliseFit(XTrain);
    //pp.NormaliseTransform(XTrain);

    // Train Network
    double alpha = 0.01;
    double lambda = 1;
    int numHidden = 50;
    int numOutput = 10;
    int Iters = 100;
    
    // One hot encode Y
    vector<vector<double>> YTrainEnc = pp.OneHotEncoding(YTrain, numOutput);
    
    // Get into Eigen format
    //from v1 to an eigen vector
    
    MatrixXd XT(XTrain.size(),XTrain[0].size());
    
    for(int i = 0; i < XTrain.size(); ++i)
    {
        vector<double> vec = XTrain[i];
        double* ptr_data = &vec[0];
        Eigen::VectorXd Xvec = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(vec.data(), vec.size());
        
        XT.row(i) = Xvec;
    }
    
    MatrixXd YT(YTrainEnc.size(),YTrainEnc[0].size());
    
    for(int i = 0; i < YTrainEnc.size(); ++i)
    {
        vector<double> vec = YTrainEnc[i];
        double* ptr_data = &vec[0];
        Eigen::VectorXd Yvec = Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(vec.data(), vec.size());
        
        YT.row(i) = Yvec;
    }
    
    NeuralNetwork nn(alpha, lambda, numHidden, numOutput, Iters);
    nn.Fit(XT, YT);
    
    cout << "Saving costs\n";
    
    // Save the costs for plotting
    vector<double> Costs = nn.GetCosts();
    string name = "/Users/samblakeman/Desktop/NNCosts.txt";
    auto filename = name.c_str();
    Utilities::SaveVectorAsCSV(Costs, filename);
    
    
    
    
    return;
}