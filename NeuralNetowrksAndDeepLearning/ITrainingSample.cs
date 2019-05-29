using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetowrksAndDeepLearning
{
    public interface ITrainingSample
    {
        double[] Input { get; }
        double[] Output { get; }
    }
}
