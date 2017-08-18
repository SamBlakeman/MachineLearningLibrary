//
//  PreProcessing.hpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 01/05/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#ifndef PreProcessing_hpp
#define PreProcessing_hpp

#include <stdio.h>
#include <vector>

using namespace std;

enum YLocation{FirstColumn,LastColumn};

class PreProcessing
{
public:
    
    // Transform features onto the same scale
    void NormaliseFit(const vector<vector<double>>& X);
    void NormaliseTransform(vector<vector<double>>& X);
    
    void StandardiseFit(const vector<vector<double>>& X);
    void StandardiseTransform(vector<vector<double>>& X);
    
    // Handle feature vector
    pair<vector<vector<double>>,vector<double>> SeperateXandY(vector<vector<double>>& FeatureVector, YLocation location);
    pair<vector<vector<vector<double>>>,vector<vector<double>>> GetTrainAndTest(const vector<vector<double>>& X, const vector<double>& Y, float trainSize);
    
    
private:
    
    vector<double> mean;
    vector<double> std;
    vector<double> min;
    vector<double> max;
    
    
};

#endif /* PreProcessing_hpp */