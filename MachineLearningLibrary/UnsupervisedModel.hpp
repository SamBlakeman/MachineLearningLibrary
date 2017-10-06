//
//  UnsupervisedModel.hpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 05/10/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#ifndef UnsupervisedModel_hpp
#define UnsupervisedModel_hpp

#include <stdio.h>
#include <vector>

using namespace std;

class UnsupervisedModel
{
public:
    
    // Fit
    virtual void Fit(const vector<vector<double>>& X) = 0;
    
    
private:
    
    
};

#endif /* UnsupervisedModel_hpp */
