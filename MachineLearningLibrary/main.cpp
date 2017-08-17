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
    
    DNNTest.Run();
    
    /*
     TODO:
     - Investigate the other activation functions, maybe try a leaky relu
     - Generative Pre-Training
     - Dropout
     - Weight decay
     - incorporate regression cost function and linear gradients at the outputs layer
     
     
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
