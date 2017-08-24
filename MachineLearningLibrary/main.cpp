//
//  main.cpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 01/05/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#include "Eigen/Dense"

#include "NormaliseTest.hpp"
#include "LRTest.hpp"
#include "LWRTest.hpp"
#include "LogRegTest.hpp"
#include "NBCTest.hpp"
#include "NeuralNetworkTest.hpp"
#include "DeepNeuralNetworkTest.hpp"

int main(int argc, const char * argv[]) {
    
    LRTest lr;
    lr.Run();
    
    
    /*
     TODO:
     
     Linear Regression:
     - Normal equations
     - Normal equations with regularization
     
     
     Logistic Regression:
     - Newtons Method
     
     
     Neural Networks:
     - R squared for regressions
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
     - Implement
     
     
     Decision Trees:
     - Implement
     
     
     AutoEncoder:
     - Implement
     
    
     Misc:
     - learning curve
     - validation curve
     
     */

    return 0;
}
