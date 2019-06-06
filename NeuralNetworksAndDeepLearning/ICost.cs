using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworksAndDeepLearning
{
    public interface ICost
    {
        double Cost(double[] activations, double[] outputs);
        double Function(double activation, double output);
        double DelCostOverDelActivation(double activation, double output);
        double DelCostOverDelWeightedInput(double weightedInput, double activation, double output);
    }
}
