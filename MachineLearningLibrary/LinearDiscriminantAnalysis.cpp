//
//  LinearDiscriminantAnalysis.cpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 19/11/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#include "LinearDiscriminantAnalysis.hpp"
#include "Utilities.hpp"

LinearDiscriminantAnalysis::LinearDiscriminantAnalysis()
{
    CostFun = CrossEntropy;
    return;
}

// Fit the classifier
void LinearDiscriminantAnalysis::Fit(const vector<vector<double>>& XTrain, const vector<double>& YTrain)
{
    
    numExamples = (int)YTrain.size();
    numFeatures = (int)XTrain[0].size();
    
    // Get the class priors
    CalculateClassPriors(YTrain);
    
    // Initialise mean and covariance matrices
    ClassMeans.assign(numClasses, vector<double> (numFeatures,0));
    CovarianceMatrix = MatrixXd::Zero(numFeatures, numFeatures);
    
    // Calculate the class means and tied covariance matrix
    CalculateClassMeans(XTrain, YTrain);
    CalculateCovarianceMatrix(XTrain, YTrain);
    
    return;
}

void LinearDiscriminantAnalysis::CalculateClassPriors(const vector<double>& YTrain)
{
    // Get the class counts
    for(int ex = 0; ex < numExamples; ++ex)
    {
        // Doesn't exist in map yet
        if(ClassCounts.find(YTrain[ex]) == ClassCounts.end())
        {
            ClassCounts.insert(make_pair(YTrain[ex], 1));
        }
        // Does exist
        else
        {
            ++ClassCounts[YTrain[ex]];
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

void LinearDiscriminantAnalysis::CalculateClassMeans(const vector<vector<double>>& XTrain, const vector<double>& YTrain)
{
    for(int ex = 0; ex < numExamples; ++ex)
    {
        ClassMeans[YTrain[ex]] = Utilities::VecAdd(ClassMeans[YTrain[ex]], XTrain[ex]);
    }
    
    for(int cl = 0; cl < numClasses; ++cl)
    {
        ClassMeans[cl] = Utilities::ScalarDiv(ClassMeans[cl], ClassCounts[cl]);
    }
    
    return;
}

void LinearDiscriminantAnalysis::CalculateCovarianceMatrix(const vector<vector<double>>& XTrain, const vector<double>& YTrain)
{
    for(int ex = 0; ex < numExamples; ++ex)
    {
        VectorXd X_i = Utilities::ConvertToEigen(XTrain[ex]);
        VectorXd mu_c = Utilities::ConvertToEigen(ClassMeans[YTrain[ex]]);
        CovarianceMatrix += (X_i - mu_c) * (X_i - mu_c).transpose();
    }
    
    CovarianceMatrix /= numExamples;
    
    return;
}




// Predict
vector<double> LinearDiscriminantAnalysis::Predict(const vector<vector<double>>& XTest)
{
    vector<double> Predictions;
    
    return Predictions;
}

void LinearDiscriminantAnalysis::SetLambda(double lambda)
{
    return;
}

void LinearDiscriminantAnalysis::SetAlpha(double alpha)
{
    return;
}

void LinearDiscriminantAnalysis::SetIterations(int iters)
{
    return;
}

void LinearDiscriminantAnalysis::SetTau(double tau)
{
    return;
}

void LinearDiscriminantAnalysis::SetC(double c)
{
    return;
}

void LinearDiscriminantAnalysis::SetVar(double var)
{
    return;
}



