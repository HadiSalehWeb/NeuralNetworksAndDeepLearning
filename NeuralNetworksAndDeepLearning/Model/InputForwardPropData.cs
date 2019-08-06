using NeuralNetworksAndDeepLearning.Interface;
using System;

namespace NeuralNetworksAndDeepLearning.Model
{
    public class InputForwardPropData : IForwardPropData
    {
        public float[] WeightedInputs => throw new InvalidOperationException("The input layer doesn't have any weighted input.");
        public float[] Activations { get; }
        public InputForwardPropData(float[] input)
        {
            Activations = input;
        }
    }
}
