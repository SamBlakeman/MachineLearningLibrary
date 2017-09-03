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
struct KFoldResults {double TrainingAccuracy; double TrainingRSquared; double TestAccuracy; double TestRSquared;};

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
    
    
protected:
    
    CostFunction CostFun;
    
private:
    
    
    
};



#endif /* MachineLearningModel_hpp */
