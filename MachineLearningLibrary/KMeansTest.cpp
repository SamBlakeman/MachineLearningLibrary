//
//  KMeansTest.cpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 01/10/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#include "KMeansTest.hpp"
#include "KMeans.hpp"
#include "Utilities.hpp"
#include <iostream>

void KMeansTest::Run()
{
    string input = "/Users/samblakeman/Desktop/ClusteringTestData.txt";
    const char* InputFile = input.c_str();
    
    vector<vector<double>> FeatureVector;
    
    FeatureVector = Utilities::ReadCSVFeatureVector(InputFile);
    
    // Create k-means object and train
    int numClusters = 3;
    int numNeighbours = 1;
    int iters = 1000;
    KMeans km(numClusters, numNeighbours, iters);
    
    km.Fit(FeatureVector);
    vector<int> AssignedCentroids = km.GetAssignedCentroids();
    
    // Save the results
    string name = "/Users/samblakeman/Desktop/AssignedCentroids.txt";
    auto filename = name.c_str();
    Utilities::SaveVectorAsCSV(AssignedCentroids, filename);
    
    return;
}