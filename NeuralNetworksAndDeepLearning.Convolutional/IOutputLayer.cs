using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworksAndDeepLearning.Convolutional
{
    public interface IOutputLayer : ILayer
    {
        float Cost(float[] previousActivations, float[] output);
        float[] GetDelCostOverDelWeightedInputs(float[] weightedInputs, float[] activations, float[] output);
    }
}
