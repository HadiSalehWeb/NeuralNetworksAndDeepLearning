using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworksAndDeepLearning
{
    public interface ICost
    {
        float Cost(float[] activations, float[] outputs);
        float Function(float activation, float output);
        float DelCostOverDelActivation(float activation, float output);
        float DelCostOverDelWeightedInput(float weightedInput, float activation, float output);
    }
}
