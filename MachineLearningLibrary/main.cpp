//
//  main.cpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 01/05/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#include "NormaliseTest.hpp"
#include "LRTest.hpp"
#include "LogRegTest.hpp"
#include "NBCTest.hpp"

int main(int argc, const char * argv[]) {
    
    
    NBCTest nbct;
    nbct.ContinuousTest();
    
    //LogRegTest lgr;
    //lgr.Test2();
    
    /*
     TODO:
     - Naive Bayes classifer
     - finish all the constructor combinations
     - make continuous and discrete an enum
     - allow for continuous input variables - needs testing
     - maybe have likelihoods calculated during fitting rather than predicting
     - accuracy, confusion matrix etc.
     - maybe also allow for strings
     
     - Neural Network - two layer implementation, forward propagation implemented
     - SVM
     - learning curve
     - validation curve
     
     */

    return 0;
}
