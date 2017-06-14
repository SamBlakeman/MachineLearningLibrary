//
//  NaiveBayesClassifier.cpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 27/05/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#include "NaiveBayesClassifier.hpp"
#include "Utilities.hpp"
#include <numeric>
#include <cmath>


// Constructors
NaiveBayesClassifier::NaiveBayesClassifier()
{
    
}

NaiveBayesClassifier::NaiveBayesClassifier(bool ContinuousInputVariables)
{
    Continuous = ContinuousInputVariables;
}

NaiveBayesClassifier::NaiveBayesClassifier(vector<double> Priors)
{
    ClassPriors = Priors;
}

NaiveBayesClassifier::NaiveBayesClassifier(double EquivalentSampSize)
{
    EquivalentSampleSize = EquivalentSampSize;
}

NaiveBayesClassifier::NaiveBayesClassifier(bool ContinuousInputVariables, vector<double> Priors)
{
    Continuous = ContinuousInputVariables;
    ClassPriors = Priors;
}

// Fit the classifier
void NaiveBayesClassifier::Fit(const vector<vector<double>>& XTrain, const vector<double>& YTrain)
{
    // Make copies for when we need to calculate likelihoods
    X = XTrain;
    Y = YTrain;
    
    numExamples = (int)YTrain.size();
    numAttributes = (int)XTrain[0].size();
    
    
    if(ClassPriors.size() == 0)
    {
        // Get the class priors
        CalculateClassPriors();
    }
    
    if(Continuous)
    {
        // Initialise mean and std matrices
        ClassAttributeMeans.assign(numClasses, vector<double> (numAttributes,0));
        ClassAttributeStds.assign(numClasses, vector<double> (numAttributes,0));
        
        CalculateClassAttributeMeansAndStds();
    }

    return;
}

void NaiveBayesClassifier::CalculateClassPriors()
{
    // Get the class counts
    for(int ex = 0; ex < numExamples; ++ex)
    {
        // Doesn't exist in map yet
        if(ClassCounts.find(Y[ex]) == ClassCounts.end())
        {
            ClassCounts.insert(make_pair(Y[ex], 1));
        }
        // Does exist
        else
        {
            ++ClassCounts[Y[ex]];
        }
    }
    
    numClasses = (int)ClassCounts.size();
    
    // Calculate the priors
    for(int c = 0; c < numClasses; ++c)
    {
        ClassPriors.push_back(ClassCounts[c]/numExamples);
    }
    
    return;
}

void NaiveBayesClassifier::CalculateClassAttributeMeansAndStds()
{
    vector<double> AttributeValues;
    double Mean = 0;
    
    // loop through each class
    for(int c=0; c < numClasses; ++c)
    {
        // loop through each attribute
        for(int a=0; a < numAttributes; ++a)
        {
            // loop through examples
            for(int ex=0; ex < numExamples; ++ex)
            {
                // if example matches current class
                if(Y[ex]==c)
                {
                    AttributeValues.push_back(X[ex][a]);
                }
            }
            
            // Set the mean and std for the class and attribute
            Mean = Utilities::GetVecMean(AttributeValues);
            ClassAttributeMeans[c][a] = Mean;
            ClassAttributeStds[c][a] = Utilities::GetVecStd(AttributeValues, Mean);
            
            AttributeValues.clear();
        }
    }
    
    return;
}

vector<double> NaiveBayesClassifier::Predict(const vector<vector<double>>& XTest)
{
    vector<double> Predictions(XTest.size());
    vector<double> ClassLikelihoods;
    
    for(int ex = 0; ex < XTest.size(); ++ex)
    {
        if(Continuous)
        {
            ClassLikelihoods = CalculateContClassLikelihoods(XTest[ex]);
        }
        else
        {
            ClassLikelihoods = CalculateDiscClassLikelihoods(XTest[ex]);
        }
        
        double map = 0;
        for(int c = 0; c < ClassLikelihoods.size(); ++c)
        {
            if((ClassLikelihoods[c]*ClassPriors[c]) > map)
            {
                map = ClassLikelihoods[c]*ClassPriors[c];
                Predictions[ex] = c;
            }
        }
    }
    
    return Predictions;
}



vector<double> NaiveBayesClassifier::CalculateContClassLikelihoods(const vector<double>& Example)
{
    double value;
    vector<double> ClassLikelihoods;
    
    // Loop through the classes
    for(int c = 0; c < numClasses; ++c)
    {
        // Loop through attributes
        double likelihood = 1;
        for(int a = 0; a < numAttributes; ++a)
        {
            value = Example[a];
            
            likelihood *= Gaussian(c, a, value);
        }
        
        ClassLikelihoods.push_back(likelihood);
    }
    
    return ClassLikelihoods;
}


vector<double> NaiveBayesClassifier::CalculateDiscClassLikelihoods(const vector<double>& Example)
{
    double value;
    vector<double> ClassLikelihoods;
    
    // Loop through the classes
    for(int c = 0; c < numClasses; ++c)
    {
        // Loop through attributes
        double likelihood = 1;
        for(int a = 0; a < numAttributes; ++a)
        {
            value = Example[a];
            double nc = GetAttributeValueCountForClass(c, a, value);
            
            likelihood *= (nc + (EquivalentSampleSize*ClassPriors[c]))/(ClassCounts[c] + EquivalentSampleSize);
        }
        
        ClassLikelihoods.push_back(likelihood);
    }
    
    return ClassLikelihoods;
}

double NaiveBayesClassifier::GetAttributeValueCountForClass(int Class, int Attribute, double Value)
{
    double nc = 0;
    for(int ex = 0; ex < numExamples; ++ex)
    {
        if(Y[ex] == Class && X[ex][Attribute] == Value)
        {
            ++nc;
        }
    }
    
    return nc;
}

double NaiveBayesClassifier::Gaussian(int Class, int Attribute, double Value)
{
    //double coeff = (1/(ClassAttributeStds[Class][Attribute]*sqrt(2*pi)));
    //double exponential = -(1/2)*(pow((Value - ClassAttributeMeans[Class][Attribute])/ClassAttributeStds[Class][Attribute], 2));
    
    double coeff = (1/(ClassAttributeStds[Class][Attribute]*sqrt(2*pi)));
    double exponential = -pow((Value-ClassAttributeMeans[Class][Attribute]),2)/(2*pow(ClassAttributeStds[Class][Attribute],2));
    
    return coeff*exp(exponential);
}

double NaiveBayesClassifier::GetAccuracy(const vector<vector<double>>& X, const vector<double>& Y)
{
    double Accuracy;
    
    vector<double> Predictions = Predict(X);
    
    double numCorrect = 0;
    double total = Y.size();
    
    for(int e = 0; e < total; ++e)
    {
        if(Predictions[e] == Y[e])
        {
            ++numCorrect;
        }
    }
    
    Accuracy = (numCorrect/total)*100;
    
    return Accuracy;
}



