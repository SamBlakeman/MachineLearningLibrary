//
//  LocallyWeightedRegression.hpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 23/08/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#ifndef LocallyWeightedRegression_hpp
#define LocallyWeightedRegression_hpp

#include <stdio.h>
#include <vector>
#include "Eigen/Dense"
#include "SupervisedModel.hpp"

using namespace std;
using namespace Eigen;

class LocallyWeightedRegression : public SupervisedModel
{
public:
    
    // Constructors
    LocallyWeightedRegression(double tau);
    
    // Fit the weights of the model
    virtual void Fit(const vector<vector<double>>& XTrain, const vector<double>& YTrain) override;
    
    // Predict
    virtual vector<double> Predict(const vector<vector<double>>& XTest) override;
    
    // Getters
    vector<double> GetWeights();
    vector<double> GetCosts();
    
    // Setters
    virtual void SetLambda(double lambda) override;
    virtual void SetAlpha(double alpha) override;
    virtual void SetIterations(int iters) override;
    virtual void SetTau(double tau) override;
    virtual void SetC(double c) override;
    virtual void SetVar(double var) override;
    
private:
    
    // Fit
    void NormalEquation(const vector<vector<double>>& XTrain, const vector<double>& YTrain);
    
    
    MatrixXd X;
    MatrixXd Y;
    
    double Tau = 0.f;
    int numFeatures = 0;
    int numTrainExamples = 0;
    
};

#endif /* LocallyWeightedRegression_hpp */
