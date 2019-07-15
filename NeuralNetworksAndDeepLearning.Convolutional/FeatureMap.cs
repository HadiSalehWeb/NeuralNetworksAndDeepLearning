using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworksAndDeepLearning.Convolutional
{
    public struct FeatureMap
    {
        public float[,] Weights { get; }
        public float Bias { get; }

        public FeatureMap(float[,] weights, float bias)
        {
            Weights = weights;
            Bias = bias;
        }
    }
}
