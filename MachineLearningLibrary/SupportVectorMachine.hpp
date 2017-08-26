//
//  SupportVectorMachine.hpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 25/08/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#ifndef SupportVectorMachine_hpp
#define SupportVectorMachine_hpp

#include <stdio.h>
#include <vector>
#include "Eigen/Dense"

enum Kernel {Linear, Gaussian};

using namespace std;
using namespace Eigen;

class SupportVectorMachine
{
public:
    
    // Constructors
    SupportVectorMachine(double c, double alpha, int iters);
    
    // Kernels
    void AddGaussianKernel(double Variance);
    
    // Fit
    void Fit(const vector<vector<double>>& X, const vector<double>& Y);
    
    // Predict
    vector<double> Predict(const vector<vector<double>>& X);
    
    // Getters
    vector<double> GetCosts();
    double GetAccuracy(const vector<vector<double>>& X, const vector<double>& Y);
    
private:
    
    // Fit
    void GradientDescent(const MatrixXd& XTrain, const MatrixXd& YTrain);
    
    double C = 1;
    double Var = 1;
    double Alpha = 0.1;
    int Iterations = 0;
    
    Kernel Ker = Linear;
    
    double numFeatures = 0;
    double numTrainExamples = 0;
    
    VectorXd Theta;
    vector<double> Costs;
    
};

#endif /* SupportVectorMachine_hpp */
