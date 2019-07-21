using System;

namespace NeuralNetworksAndDeepLearning.Convolutional
{
    public abstract class FullyConnectedBase : ILayer
    {
        protected float[,] WeightMatrix { get; set; }
        public int ParameterCount { get; private set; }
        public int OutputDimension { get; }
        public int InputDimension { get; private set; }

        public FullyConnectedBase(int outputDimension)
        {
            OutputDimension = outputDimension;
        }

        public void Initialize(int outputDimensionOfPreviousLayer)
        {
            ParameterCount = OutputDimension * (outputDimensionOfPreviousLayer + 1);
            InputDimension = outputDimensionOfPreviousLayer;
            WeightMatrix = new float[OutputDimension, outputDimensionOfPreviousLayer + 1];

            var std = 1 / (float)Math.Sqrt(outputDimensionOfPreviousLayer);
            var rand = new Random();

            for (int i = 0; i < OutputDimension; i++)
                for (int j = 0; j < outputDimensionOfPreviousLayer + 1; j++)
                    WeightMatrix[i, j] = j == outputDimensionOfPreviousLayer ? 0 : MLMath.Gaussian(rand, 0, std);
        }

        public float[] GetWeightedInput(float[] input)
        {
            float[] ret = new float[OutputDimension];

            for (int i = 0; i < OutputDimension; i++)
            {
                ret[i] = 0f;

                for (int j = 0; j < InputDimension; j++)
                    ret[i] += input[j] * WeightMatrix[i, j];

                ret[i] += WeightMatrix[i, InputDimension];
            }

            return ret;
        }

        public float[] Feedforward(float[] input)
        {
            return GetActivation(GetWeightedInput(input));
        }

        public void UpdateParameters(float[] costGradient)
        {
            for (int i = 0, c = 0; i < OutputDimension; i++)
                for (int j = 0; j < InputDimension + 1; j++, c++)
                    WeightMatrix[i, j] += costGradient[c];
        }

        public abstract float[] GetActivation(float[] weightedInput);
    }
}
