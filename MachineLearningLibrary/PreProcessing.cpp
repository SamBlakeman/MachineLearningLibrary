//
//  PreProcessing.cpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 01/05/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#include "PreProcessing.hpp"
#include "Utilities.hpp"
#include <cmath>
#include <algorithm>


void PreProcessing::NormaliseFit(const vector<vector<double>>& X)
{
    // loop through the features
    for(int feature = 0; feature < X[0].size(); ++feature)
    {
        min.push_back(X[0][feature]);
        max.push_back(X[0][feature]);
        
        // get the min and the max over all the examples
        for(int example = 0; example < X.size(); ++example)
        {
            if(X[example][feature] > max[feature]){max[feature] = X[example][feature];}
            if(X[example][feature] < min[feature]){min[feature] = X[example][feature];}
        }
    }
    return;
}

void PreProcessing::NormaliseTransform(vector<vector<double>>& X)
{
    // loop through the features
    for(int feature = 0; feature < X[0].size(); ++feature)
    {
        // normalise
        for(int example = 0; example < X.size(); ++example)
        {
            X[example][feature] = (X[example][feature] - min[feature])/(max[feature] - min[feature]);
        }
    }
    return;
}



void PreProcessing::StandardiseFit(const vector<vector<double>>& X)
{
    // loop through the features
    for(int feature = 0; feature < X[0].size(); ++feature)
    {
        // get the mean
        double m = 0;
        for(int example = 0; example < X.size(); ++example)
        {
            m += X[example][feature];
        }
        mean.push_back(m/X.size());
        
        // get the devations from the mean
        double var = 0;
        for(int example = 0; example < X.size(); ++example)
        {
            var += pow((X[example][feature] - m),2);
        }
        var = var/X.size();
        std.push_back(sqrt(var));
        
    }
    return;
}



void PreProcessing::StandardiseTransform(vector<vector<double>>& X)
{
    // loop through the features
    for(int feature = 0; feature < X[0].size(); ++feature)
    {
        // standardise
        for(int example = 0; example < X.size(); ++example)
        {
            X[example][feature] = (X[example][feature] - mean[feature])/std[feature];
        }
    }
    return;
}



pair<vector<vector<double>>,vector<double>> PreProcessing::SeperateXandY(vector<vector<double>>& FeatureVector, YLocation location)
{
    random_shuffle (FeatureVector.begin(), FeatureVector.end());
    
    int numExamples = (int)FeatureVector.size();
    int numFeatures = (int)FeatureVector[0].size() - 1;
    vector<vector<double>> X(numExamples, vector<double>(numFeatures,0));
    vector<double> Y(numExamples,0);
    
    // Seperate out feature vector into predictors (x) and outcomes (y)
    
    if(location == LastColumn)
    {
        for(int e = 0; e < numExamples; ++e)
        {
            for(int f = 0; f < numFeatures; ++f)
            {
                X[e][f] = FeatureVector[e][f];
            }
        
            Y[e] = FeatureVector[e][numFeatures];
        }
    }
    else if (location == FirstColumn)
    {
        for(int e = 0; e < numExamples; ++e)
        {
            for(int f = 1; f < numFeatures + 1; ++f)
            {
                X[e][f] = FeatureVector[e][f];
            }
            
            Y[e] = FeatureVector[e][0];
        }
    }

    return make_pair(X, Y);
}



pair<vector<vector<vector<double>>>,vector<vector<double>>> PreProcessing::GetTrainAndTest(const vector<vector<double>>& X, const vector<double>& Y, float trainSize)
{
    int numExamples = (int)X.size();
    int numFeatures = (int)X[0].size();
    int numExamplesTrain = ceil(numExamples*trainSize);
    
    vector<vector<double>> XTrain;
    vector<double> YTrain(numExamplesTrain);
    vector<vector<double>> XTest;
    vector<double> YTest(numExamples-numExamplesTrain);
    vector<double> E(X[0].size());
    
    // Create the training set
    for(int example = 0; example < numExamplesTrain; ++example)
    {
        for(int feature = 0; feature < numFeatures; ++feature)
        {
            E[feature] = (X[example][feature]);
        }
        
        XTrain.push_back(E);
        
        YTrain[example] = Y[example];
    }
    
    // Create the test set
    for(int example = numExamplesTrain; example < numExamples; ++example)
    {
        for(int feature = 0; feature < numFeatures; ++feature)
        {
            E[feature] = (X[example][feature]);
        }
        
        XTest.push_back(E);
        
        YTest[example-numExamplesTrain] = Y[example];
    }
    
    vector<vector<vector<double>>> Xs = {XTrain, XTest};
    vector<vector<double>> Ys = {YTrain, YTest};
    
    return make_pair(Xs,Ys);
}


vector<vector<double>> PreProcessing::OneHotEncoding(vector<double>& Y, int numOut)
{
    int numExamples = (int)Y.size();
    vector<vector<double>> EncodedY (numExamples, vector<double>(numOut, 0));
    
    for(int e = 0; e < numExamples; ++e)
    {
        EncodedY[e][Y[e]] = 1;
    }
    
    return EncodedY;
}