//
//  NaiveBayesClassifier.hpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 27/05/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#ifndef NaiveBayesClassifier_hpp
#define NaiveBayesClassifier_hpp

#include <stdio.h>
#include <vector>
#include <map>

using namespace std;

class NaiveBayesClassifier
{
public:
    
    // Constructors
    NaiveBayesClassifier();
    NaiveBayesClassifier(bool ContinuousInputVariables);
    NaiveBayesClassifier(double EquivalentSampSize);
    NaiveBayesClassifier(vector<double> Priors);
    NaiveBayesClassifier(bool ContinuousInputVariables, vector<double> Priors);
    
    // Fit the classifier
    void Fit(const vector<vector<double>>& XTrain, const vector<double>& YTrain);
    
    // Predict
    vector<double> Predict(const vector<vector<double>>& XTest);
    
    // Getters
    double GetAccuracy(const vector<vector<double>>& X, const vector<double>& Y);
    
    
private:
    
    // Fit
    void CalculateClassPriors();
    void CalculateClassAttributeMeansAndStds();
    
    // Predict
    vector<double> CalculateDiscClassLikelihoods(const vector<double>& Example);
    vector<double> CalculateContClassLikelihoods(const vector<double>& Example);
    double GetAttributeValueCountForClass(int Class, int Attribute, double Value);
    double Gaussian(int Class, int Attribute, double Value);
    
    bool Continuous = false;
    int numClasses = 2;
    int numExamples;
    int numAttributes;
    double EquivalentSampleSize = 0;
    
    vector<vector<double>> X;
    vector<double> Y;
    
    map<int,double> ClassCounts;
    vector<double> ClassPriors;
    vector<vector<double>> ClassAttributeMeans;
    vector<vector<double>> ClassAttributeStds;
    
    const double pi = 3.14159265358979323846;
    
};

#endif /* NaiveBayesClassifier_hpp */