using System;
using System.Linq;

namespace NeuralNetworksAndDeepLearning.Convolutional
{
    public class FullyConnectedOutput : FullyConnected, IOutputLayer
    {
        public abstract class CostFunction
        {
            internal abstract (Func<float, float>, Func<float, float>) Activation { get; }
            internal abstract float Calculate(float[] activations, float[] output);
            internal abstract float[] Error(float[] output, float[] activations, float[] weightedInputs);
            public static CostFunction CrossEntropy => new CrossEntropy();
            public static CostFunction QuadraticCost((Func<float, float>, Func<float, float>) activation) => new QuadraticCost(activation);
        }

        private class CrossEntropy : CostFunction
        {
            internal override (Func<float, float>, Func<float, float>) Activation => Activations.Sigmoid;
            public CrossEntropy() { }

            internal override float Calculate(float[] activations, float[] output)
            {
                return (float)Enumerable.Range(0, activations.Length)
                    .Aggregate(0.0, (a, c) => a - output[c] * Math.Log(activations[c]) - (1 - output[c]) * Math.Log(1 - activations[c]));
            }

            internal override float[] Error(float[] output, float[] activations, float[] weightedInputs)
            {
                return activations.Select((a, i) => a - output[i]).ToArray();
            }
        }

        private class QuadraticCost : CostFunction
        {
            internal override (Func<float, float>, Func<float, float>) Activation { get; }
            private Func<float, float> ActivationDerivative => Activation.Item2;
            public QuadraticCost((Func<float, float>, Func<float, float>) activation)
            {
                Activation = activation;
            }

            internal override float Calculate(float[] activations, float[] output)
            {
                return Enumerable.Range(0, activations.Length)
                    .Aggregate(0f, (a, c) => a + (activations[c] - output[c]) * (activations[c] - output[c]));
            }

            internal override float[] Error(float[] output, float[] activations, float[] weightedInputs)
            {
                return activations.Select((a, i) => (a - output[i]) * ActivationDerivative(weightedInputs[i])).ToArray();
            }
        }

        private readonly CostFunction cost;

        public FullyConnectedOutput(int outputDimension, CostFunction cost) : base(outputDimension, cost.Activation)
        {
            this.cost = cost;
        }

        public float Cost(float[] previousActivations, float[] output)
        {
            return cost.Calculate(Feedforward(previousActivations), output);
        }

        public float[] GetError(float[] output, float[] activations, float[] weightedInputs)
        {
            return cost.Error(output, activations, weightedInputs);
        }

        public float[] BackpropagateParameters(float[] error, float[] inActivations)
        {
            float[] gradient = new float[WeightMatrix.GetLength(0) * WeightMatrix.GetLength(1)];
            for (int i = 0; i < OutputDimension; i++)
            {
                var index = i * (InputDimension + 1);

                for (int j = 0; j < InputDimension; j++)
                    gradient[index + j] = error[i] * inActivations[j];

                gradient[index + InputDimension] = error[i];
            }

            return gradient;
        }

        public float[] BackpropagateErrorToActivation(float[] error)
        {
            float[] del = new float[InputDimension];

            for (int i = 0; i < InputDimension; i++)
                for (int j = 0; j < OutputDimension; j++)
                    del[i] += error[j] * WeightMatrix[j, i];

            return del;
        }
    }
}
