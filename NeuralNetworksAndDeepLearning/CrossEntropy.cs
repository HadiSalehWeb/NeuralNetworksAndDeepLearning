using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworksAndDeepLearning
{
    public class CrossEntropy : ICost
    {
        public float Cost(float[] activations, float[] outputs)
        {
            float cost = 0f;

            for (int i = 0; i < activations.Length; i++)
                cost += Function(activations[i], outputs[i]);

            return cost;
        }

        public float Function(float activation, float output)
        {
            return (float)(-output * Math.Log(activation) - (1 - output) * Math.Log(1 - activation));
        }

        public float DelCostOverDelActivation(float activation, float output)
        {
            return (activation - output) / (activation * (1 - activation));
        }

        public float DelCostOverDelWeightedInput(float weightedInput, float activation, float output)
        {
            return (activation - output);
        }
    }
}
