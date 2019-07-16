using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetworksAndDeepLearning.Convolutional
{
    public class FullyConnectedOutput : FullyConnected, IOutputLayer
    {
        public abstract class CostFunction
        {
            internal abstract ActivationFunctionType Activation { get; }
            internal abstract float Calculate(float[] activations, float[] output);
            public static CostFunction CrossEntropy => new CrossEntropy();
            public static CostFunction QuadraticCost(ActivationFunctionType activation) => new QuadraticCost(activation);
            internal abstract float[] Error(float[] output, float[] activations, float[] weightedInputs);
        }

        private class CrossEntropy : CostFunction
        {
            internal override ActivationFunctionType Activation => ActivationFunctionType.Sigmoid;
            public CrossEntropy() { }

            internal override float Calculate(float[] activations, float[] output)
            {
                return (float)Enumerable.Range(0, activations.Length)
                    .Aggregate(0.0, (a, c) => a - output[c] * Math.Log(activations[c]) - (1 - output[c]) * Math.Log(1 - activations[c]));
            }
        }

        private class QuadraticCost : CostFunction
        {
            internal override ActivationFunctionType Activation { get; }
            public QuadraticCost(ActivationFunctionType activation)
            {
                Activation = activation;
            }

            internal override float Calculate(float[] activations, float[] output)
            {
                return Enumerable.Range(0, activations.Length)
                    .Aggregate(0f, (a, c) => a + (activations[c] - output[c]) * (activations[c] - output[c]));
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
            throw new NotImplementedException();
        }
    }
}
