//
//  MachineLearningModel.hpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 02/09/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#ifndef MachineLearningModel_hpp
#define MachineLearningModel_hpp

#include <stdio.h>
#include <vector>
#include "Eigen/Dense"

using namespace std;
using namespace Eigen;

enum CostFunction {CrossEntropy, SumOfSquaredErrors};
enum Parameter {Lambda, Alpha, Iterations, Tau, C, Var};
struct KFoldResults {double TrainingAccuracy; double TrainingRSquared; double TestAccuracy; double TestRSquared;};
struct ValidationCurveResults {vector<double> TrainMeanAccuracy; vector<double> TestMeanAccuracy;};

class MachineLearningModel
{
public:
    
    // Fit
    virtual void Fit(const vector<vector<double>>& X, const vector<double>& Y) = 0;
    
    // Predict
    virtual vector<double> Predict(const vector<vector<double>>& XTest) = 0;
    
    // Goodness of fit
    double CalculateRSquared(const vector<vector<double>>& X, const vector<double>& Y);
    double CalculateAccuracy(const vector<vector<double>>& X, const vector<double>& Y);
    
    // Cross validation
    KFoldResults KFoldCrossValidation(const vector<vector<double>>& X, const vector<double>& Y, int numFolds);
    
    // Validation curve
    ValidationCurveResults ValidationCurve(const vector<vector<double>>& X, const vector<double>& Y, Parameter Param, vector<double> ParamRange, int numFolds);
    
    // Setters
    virtual void SetLambda(double lambda) = 0;
    virtual void SetAlpha(double alpha) = 0;
    virtual void SetIterations(int iters) = 0;
    virtual void SetTau(double tau) = 0;
    virtual void SetC(double c) = 0;
    virtual void SetVar(double var) = 0;
    
protected:
    
    CostFunction CostFun;
    
private:
    
    
    
};



#endif /* MachineLearningModel_hpp */
