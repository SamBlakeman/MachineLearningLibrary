//
//  KMeans.cpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 01/10/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#include "KMeans.hpp"
#include <cstdlib>
#include "Eigen/Dense"
#include "Utilities.hpp"

KMeans::KMeans(int numClusters, int numNeighbours, int iters)
{
    numCent = numClusters;
    numNeigh = numNeighbours;
    Iterations = iters;
}

void KMeans::Fit(const vector<vector<double>>& X)
{
    numFeatures = int(X[0].size());
    numExamples = int(X.size());

    Centroids = vector<vector<double>>(numCent,vector<double>(numFeatures,0));
    
    // K-means++ initialisation
    KMeansPlusPlus(X);
    
    // Run k-means clustering
    Cluster(X);
    
    return;
}

void KMeans::KMeansPlusPlus(const vector<vector<double>>& X)
{
    srand (time(NULL));
    int first = rand() % numExamples;
    Centroids.push_back(X[first]);
    
    vector<double> DistanceToNearestCentroid(numExamples,0);
    double ShortestDistance = 0;
    
    for(int k = 1; k < numCent; ++k)
    {
        // Calculate the distance to the nearest centroid for each example
        for(int example = 0; example < numExamples; ++example)
        {
            VectorXd x = Utilities::ConvertToEigen(X[example]);
            
            for(int cent = 0; cent < Centroids.size(); ++cent)
            {
                VectorXd u = Utilities::ConvertToEigen(Centroids[cent]);
                
                double Distance = (x - u).dot(x - u);
                
                if(cent == 0)
                {
                    ShortestDistance = Distance;
                }
                else if(Distance < ShortestDistance)
                {
                    ShortestDistance = Distance;
                }
            }
            
            DistanceToNearestCentroid[example] = ShortestDistance;
        }
        
        // Set the example with the largest distance to the nearest centroid as the next centroid
        int NewCentroid = 0;
        double LargestDistance = DistanceToNearestCentroid[0];
        
        for(int example = 0; example < numExamples; ++example)
        {
            if(DistanceToNearestCentroid[example] > LargestDistance)
            {
                LargestDistance = DistanceToNearestCentroid[example];
                NewCentroid = example;
            }
        }
        
        Centroids.push_back(X[NewCentroid]);
    
    }
    
    return;
}

void KMeans::Cluster(const vector<vector<double>>& X)
{
    double ShortestDistance = 0;
    vector<int> AssignedCentroids(numExamples,0);
    
    for(int iter = 0; iter < Iterations; ++iter)
    {
        // Assign examples to the nearest centroid
        for(int example; example < numExamples; ++example)
        {
            VectorXd x = Utilities::ConvertToEigen(X[example]);
            
            for(int cent = 0; cent < numCent; ++cent)
            {
                VectorXd u = Utilities::ConvertToEigen(Centroids[cent]);
                
                double Distance = (x - u).dot(x - u);
                
                if(cent == 0)
                {
                    ShortestDistance = Distance;
                    AssignedCentroids[example] = cent;
                }
                else if(Distance < ShortestDistance)
                {
                    ShortestDistance = Distance;
                    AssignedCentroids[example] = cent;
                }
            }
            
        }
        
        // Update the centroids
        vector<double> numMembers(numCent,0);
        
        // Count the number of members for each centroid
        for(int example; example < numExamples; ++example)
        {
            ++numMembers[AssignedCentroids[example]];
        }
        
        Centroids = vector<vector<double>>(numCent,vector<double>(numFeatures,0));
        
        // Add up the examples for each centroid
        for(int example; example < numExamples; ++example)
        {
            Centroids[AssignedCentroids[example]] = Utilities::VecAdd(Centroids[AssignedCentroids[example]], X[example]);
        }
        
        // Divide by the number of members to get the mean vector
        for(int cent; cent < numCent; ++cent)
        {
            Centroids[cent] = Utilities::ScalarDiv(Centroids[cent], numMembers[cent]);
        }
        
    }
    
    return;
}

void KMeans::Predict(const vector<vector<double>>& X)
{
    
    
    return;
}