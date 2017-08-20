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
#include "LogRegTest.hpp"
#include "NBCTest.hpp"
#include "NeuralNetworkTest.hpp"
#include "DeepNeuralNetworkTest.hpp"

int main(int argc, const char * argv[]) {
    
    NeuralNetworkTest NNTest;
    DeepNeuralNetworkTest DNNTest;
    NBCTest nbc;
    
    nbc.DiscreteTest();
    //DNNTest.RunClassificationTest();
    //DNNTest.RunRegressionTest();
    //NNTest.Run();
    
    /*
     TODO:
     - sort out the laplace smoothin, just have +1 on the top and +k on the bottom, quivalent sample size is too confusing because it would need to be specified for each attribute etc.
     
     - Regression for standard neural network
     - Generative Pre-Training
     - Dropout
     - Weight decay
     - Softmax
     
     
     - Naive Bayes classifer
     - finish all the constructor combinations
     - make continuous and discrete an enum
     - allow for continuous input variables - needs testing
     - maybe have likelihoods calculated during fitting rather than predicting
     - accuracy, confusion matrix etc.
     - maybe also allow for strings
     
     - SVM
     - learning curve
     - validation curve
     
     */

    return 0;
}
