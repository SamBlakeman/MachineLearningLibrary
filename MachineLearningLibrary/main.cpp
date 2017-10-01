//
//  main.cpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 01/05/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#include "Eigen/Dense"

#include "NormaliseTest.hpp"
#include "KFoldTest.hpp"
#include "LRTest.hpp"
#include "LWRTest.hpp"
#include "LogRegTest.hpp"
#include "SVMTest.hpp"
#include "NBCTest.hpp"
#include "NeuralNetworkTest.hpp"
#include "DeepNeuralNetworkTest.hpp"
#include "ValidationCurveTest.hpp"
#include "LearningCurveTest.hpp"

int main(int argc, const char * argv[]) {
    
    LearningCurveTest LCTest;
    LCTest.Run();
    
    /*
     TODO:
     
     K-Means:
     - implement
     - use k-means++ initialization
     
     Gaussian Mixture Model:
     - implement
     
     K-Nearest Neighbours:
     - implement
     
     Machine Learning Model Base Class:
     - learning curve
     - ROC or TPR, FPR etc.
     - other useful exploratory methods that are shared for all models
     - other subclasses?
     
     
     Logistic Regression:
     - Newtons Method --> unsure of how to calculate the Hessian
     
     
     Neural Networks:
     - Allow for several outputs for regression (sort out one hot encoding function)
     - Generative Pre-Training
     - Dropout
     - Weight decay
     - Softmax
     
     
     Naive Bayes classifer:
     - finish all the constructor combinations
     - make continuous and discrete an enum
     - allow for continuous input variables - needs testing
     - maybe have likelihoods calculated during fitting rather than predicting
     - accuracy, confusion matrix etc.
     - maybe also allow for strings
     
     
     Gaussian Discriminant Analysis:
     - Implement
     
     
     SVM:
     - check what y should be!! should it be -1 rather than 0?
     
     
     Decision Trees:
     - Implement
     
     
     AutoEncoder:
     - Implement
     
     
     */

    return 0;
}
