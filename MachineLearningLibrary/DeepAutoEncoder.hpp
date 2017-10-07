//
//  DeepAutoEncoder.hpp
//  MachineLearningLibrary
//
//  Created by Sam Blakeman on 06/10/2017.
//  Copyright © 2017 Sam Blakeman. All rights reserved.
//

#ifndef DeepAutoEncoder_hpp
#define DeepAutoEncoder_hpp

#include <stdio.h>
#include "UnsupervisedModel.hpp"
#include "NeuralNetworkModel.hpp"

class DeepAutoEncoder : public UnsupervisedModel, public NeuralNetworkModel
{
public:
    
    // Construction
    DeepAutoEncoder(double alpha, double lambda, int Iters);
    
    // Fit
    virtual void Fit(const vector<vector<double>>& X) override;
    
    // Encode
    vector<vector<double>> GetEncodedLayer (const vector<vector<double>>& X, int LayerToReadOut);
    
    
private:
    
    
    
};

#endif /* DeepAutoEncoder_hpp */
