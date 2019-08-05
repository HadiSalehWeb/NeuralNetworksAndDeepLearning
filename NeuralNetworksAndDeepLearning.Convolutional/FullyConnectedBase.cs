using System;
using System.Linq;

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

            var stdv = 1 / (float)Math.Sqrt(outputDimensionOfPreviousLayer);
            var rand = new Random();

            for (int i = 0; i < OutputDimension; i++)
                for (int j = 0; j < outputDimensionOfPreviousLayer + 1; j++)
                    WeightMatrix[i, j] = MLMath.Gaussian(rand, 0, j == outputDimensionOfPreviousLayer ? 1 : stdv);
        }

        public float[] GetWeightedInput(float[] input)
        {
            float[] weightedInput = new float[OutputDimension];

            for (int i = 0; i < OutputDimension; i++)
            {
                weightedInput[i] = 0f;

                for (int j = 0; j < InputDimension; j++)
                    weightedInput[i] += input[j] * WeightMatrix[i, j];

                weightedInput[i] += WeightMatrix[i, InputDimension];
            }

            if (weightedInput.Any(float.IsInfinity)) throw new Exception("Yo what the fuck");

            return weightedInput;
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
