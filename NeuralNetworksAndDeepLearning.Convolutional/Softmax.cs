using System;
using System.Linq;

namespace NeuralNetworksAndDeepLearning.Convolutional
{
    //Todo: is FullyConnectedBase's weightInit actually a valid way of initializing weights for a softmax layer? Revise the NNADL section and think about it
    public class Softmax : FullyConnectedBase, IOutputLayer
    {
        public Softmax(int outputDimension) : base(outputDimension) { }

        public override float[] GetActivation(float[] weightedInput)
        {
            var exp = weightedInput.Select(z => Math.Exp(z));
            var sum = exp.Sum();
            return exp.Select(ez => (float)(ez / sum)).ToArray();
        }

        public float Cost(float[] previousActivations, float[] output)
        {
            var activations = Feedforward(previousActivations);
            return (float)Enumerable.Range(0, output.Length).Aggregate(0.0, (a, c) => a - output[c] * Math.Log(activations[c]) - (1 - output[c]) * Math.Log(1 - activations[c]));
        }

        public float[] GetError(float[] output, float[] activations, float[] weightedInputs)
        {
            return activations.Select((a, i) => a - output[i]).ToArray();
        }

        public float[] BackpropagateParameters(float[] error, float[] inActivations)
        {
            float[] ret = new float[WeightMatrix.GetLength(0) * WeightMatrix.GetLength(1)];
            for (int i = 0; i < OutputDimension; i++)
            {
                var index = i * (InputDimension + 1);

                for (int j = 0; j < InputDimension; j++)
                    ret[index + j] = error[i] * inActivations[j];

                ret[index + InputDimension] = error[i];
            }

            return ret;
        }

        public float[] BackpropagateErrorToActivation(float[] error)
        {
            float[] ret = new float[InputDimension];

            for (int i = 0; i < InputDimension; i++)
                for (int j = 0; j < OutputDimension; j++)
                    ret[i] += error[j] * WeightMatrix[j, i];

            return ret;
        }
    }
}
