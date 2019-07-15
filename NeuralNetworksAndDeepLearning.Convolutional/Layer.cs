using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworksAndDeepLearning.Convolutional
{
    public delegate float ActivationFunction(float x);

    public interface ILayer
    {
        int WeightCount { get; }
        float[] Feedforward(float[] input);
        float[] GetWeightedInput(float[] input);
        float[] GetActivation(float[] weightedInput);
        void Initialize(int previousLayerWeightCount);
        void UpdateWeights(float[] costGradient);
        float[] BackpropagateWeights(float[] delCostOverDelWeightedInputs);
        float[] BackpropagateError(float[] delCostOverDelWeightedInputs);
    }
}
