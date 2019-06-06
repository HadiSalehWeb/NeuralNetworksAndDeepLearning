using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworksAndDeepLearning
{
    public class CrossEntropy : ICost
    {
        public double Cost(double[] activations, double[] outputs)
        {
            double cost = 0.0;

            for (int i = 0; i < activations.Length; i++)
                cost += Function(activations[i], outputs[i]);

            return cost;
        }

        public double Function(double activation, double output)
        {
            return -output * Math.Log(activation) - (1 - output) * Math.Log(1 - activation);
        }

        public double DelCostOverDelActivation(double activation, double output)
        {
            return (activation - output) / (activation * (1 - activation));
        }

        public double DelCostOverDelWeightedInput(double weightedInput, double activation, double output)
        {
            return (activation - output);
        }
    }
}
