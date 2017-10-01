//
//  KMeans.hpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 01/10/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#ifndef KMeans_hpp
#define KMeans_hpp

#include <stdio.h>
#include <vector>

using namespace std;

class KMeans
{
public:
    
    // Constructors
    KMeans(int numClusters, int numNeighbours, int iters);
    
    // Fit
    void Fit(const vector<vector<double>>& X);
    
    // Predict
    void Predict(const vector<vector<double>>& X);
    
    // Getters
    vector<int> GetAssignedCentroids();
    vector<vector<double>> GetCentroids();
    vector<double> GetDistortions();
    
private:
    
    void KMeansPlusPlus(const vector<vector<double>>& X);
    void Cluster(const vector<vector<double>>& X);
    
    int numCent;
    int numNeigh;
    int Iterations;
    int numFeatures;
    int numExamples;
    vector<vector<double>> Centroids;
    vector<int> AssignedCentroids;
    vector<double> Distortions;
    
};

#endif /* KMeans_hpp */
