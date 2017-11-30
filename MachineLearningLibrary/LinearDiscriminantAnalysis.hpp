//
//  LinearDiscriminantAnalysis.hpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 19/11/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#ifndef LinearDiscriminantAnalysis_hpp
#define LinearDiscriminantAnalysis_hpp

#include <stdio.h>
#include <map>
#include <vector>
#include "SupervisedModel.hpp"

using namespace std;
using namespace Eigen;

class LinearDiscriminantAnalysis : public SupervisedModel
{
    
public:
    
    // Constructors
    LinearDiscriminantAnalysis();
    
    // Fit the classifier
    virtual void Fit(const vector<vector<double>>& XTrain, const vector<double>& YTrain) override;
    
    // Predict
    virtual vector<double> Predict(const vector<vector<double>>& XTest) override;
    
    // Getters
    
    // Setters
    virtual void SetLambda(double lambda) override;
    virtual void SetAlpha(double alpha) override;
    virtual void SetIterations(int iters) override;
    virtual void SetTau(double tau) override;
    virtual void SetC(double c) override;
    virtual void SetVar(double var) override;
    
    
private:
    
    // Fit
    void CalculateClassPriors(const vector<double>& YTrain);
    void CalculateClassMeans(const vector<vector<double>>& XTrain, const vector<double>& YTrain);
    void CalculateCovarianceMatrix(const vector<vector<double>>& XTrain, const vector<double>& YTrain);
    
    // Predict
    vector<double> CalculateClassConditionalDensities(const vector<double>& Example);
    double MultivariateGaussian(int Class, int Attribute, double Value);
    
    int numClasses;
    int numExamples;
    int numFeatures;
    
    vector<vector<double>> X;
    vector<double> Y;
    
    map<int,double> ClassCounts;
    vector<double> ClassPriors;
    vector<vector<double>> ClassMeans;
    MatrixXd CovarianceMatrix;
    
    const double pi = 3.14159265358979323846;
    
};

#endif /* LinearDiscriminantAnalysis_hpp */
