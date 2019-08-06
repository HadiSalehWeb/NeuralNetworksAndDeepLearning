using System;
using System.Linq;
using NeuralNetworksAndDeepLearning.Interface;
using NeuralNetworksAndDeepLearning.Model;

namespace NeuralNetworksAndDeepLearning.Layer
{
    //Todo: is FullyConnectedBase's weightInit actually a valid way of initializing weights for a softmax layer? Revise the NNADL section and think about it
    public class Softmax : FullyConnectedBase, IOutputLayer
    {
        public Softmax(int outputDimension) : base(outputDimension) { }

        public override float[] GetActivation(float[] weightedInput)
        {
            var maxW = weightedInput.Max();
            var exp = weightedInput.Select(z => Math.Exp(z - maxW));
            var sum = exp.Sum();
            var ret = exp.Select(ez => (float)(ez / sum)).ToArray();
            if (ret.Any(float.IsNaN)) throw new Exception("Yo what the fuck");
            return ret;
        }

        public float Cost(float[] previousActivations, float[] output)
        {
            var activations = Feedforward(previousActivations);
            return (float)Enumerable.Range(0, output.Length).Aggregate(0.0, (a, c) => a - output[c] * Math.Log(activations[c]) - (1 - output[c]) * Math.Log(1 - activations[c]));
        }

        public float[] BackpropagateErrorToActivation(float[] error)
        {
            float[] del = new float[InputDimension];

            for (int i = 0; i < InputDimension; i++)
                for (int j = 0; j < OutputDimension; j++)
                    del[i] += error[j] * WeightMatrix[j, i];

            return del;
        }

        public float[] GetError(float[] output, IForwardPropData ownForwardPropData)
        {
            return ownForwardPropData.Activations.Select((a, i) => a - output[i]).ToArray();
        }

        public float[] Backprop(float[] error, IForwardPropData ownForwardPropData)
        {
            float[] gradient = new float[WeightMatrix.GetLength(0) * WeightMatrix.GetLength(1)];
            for (int i = 0; i < OutputDimension; i++)
            {
                var index = i * (InputDimension + 1);

                for (int j = 0; j < InputDimension; j++)
                    gradient[index + j] = error[i] * ownForwardPropData.Activations[j];

                gradient[index + InputDimension] = error[i];
            }

            return gradient;
        }
    }
}
