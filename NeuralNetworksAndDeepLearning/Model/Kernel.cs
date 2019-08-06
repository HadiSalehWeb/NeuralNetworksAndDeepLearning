using NeuralNetworksAndDeepLearning.Util;
using System;

namespace NeuralNetworksAndDeepLearning.Model
{
    public struct Kernel
    {
        public float[,,] Weights { get; }
        public float Bias { get; set; }

        public Kernel(int depth, int width, int height) : this(depth, width, height, new Random(DateTime.Now.ToString().GetHashCode())) { }
        public Kernel(int depth, int width, int height, Random rand)
        {
            Weights = new float[depth, width, height];
            var stdv = 1 / (float)Math.Sqrt(depth * width * height);

            for (int l = 0; l < depth; l++)
                for (int i = 0; i < width; i++)
                    for (int j = 0; j < height; j++)
                        Weights[l, i, j] = MLMath.Gaussian(rand, 0, stdv);

            Bias = MLMath.Gaussian(rand, 0, 1);
        }

        public Kernel(float[,,] weights, float bias)
        {
            Weights = weights;
            Bias = bias;
        }

        internal float Filter(float[] input, int n, int m, int inputWidth, int inputHeight)
        {
            float weightedInput = 0;

            for (int i = 0; i < Weights.GetLength(0); i++)
                for (int j = 0; j < Weights.GetLength(1); j++)
                    for (int k = 0; k < Weights.GetLength(2); k++)
                        weightedInput += Weights[i, j, k] * input[i * inputWidth * inputWidth + (j + n) * inputHeight + (i + m)];

            return weightedInput;
        }
    }
}
