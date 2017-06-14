//
//  NormaliseTest.cpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 01/05/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#include "NormaliseTest.hpp"
#include "PreProcessing.hpp"
#include "Utilities.hpp"
#include <vector>
#include <iostream>

using namespace std;

void NormaliseTest::run()
{
    vector<double> ExampleOne = {1,0,10,0};
    vector<double> ExampleTwo = {2,.5,20,2};
    vector<double> ExampleThree = {3,1,30,10};
    vector<vector<double>> X = {ExampleOne,ExampleTwo,ExampleThree};

    PreProcessing PP;

    // Normalise
    /*vector<vector<double>> X_norm = PP.Normalise(X);
    Utilities::Print2DVector(X_norm);
    cout << endl;*/
    
    // Standardize
    PP.StandardiseFit(X);
    PP.StandardiseTransform(X);
    Utilities::Print2DVector(X);
    cout << endl;
    
    // Split feature vector
    /*pair<vector<vector<double>>,vector<vector<double>>> FeatureVectorSplit = PP.SplitFeatureVector(X_norm, 0.6);
    vector<vector<double>> XTrain = FeatureVectorSplit.first;
    vector<vector<double>> XTest = FeatureVectorSplit.second;
    
    //Utilities::Print2DVector(XTrain);
    cout << endl;
    
    Utilities::Print2DVector(XTest);
    cout << endl;*/
    
    return;
}