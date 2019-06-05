using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworksAndDeepLearning
{
    public struct TrainingSample
    {
        public double[] Input { get; }
        public double[] Output { get; }

        public TrainingSample(double[] input, double[] output)
        {
            Input = input;
            Output = output;
        }

        public TrainingSample(double input, double output)
        {
            Input = new double[] { input };
            Output = new double[] { output };
        }
    }
}
