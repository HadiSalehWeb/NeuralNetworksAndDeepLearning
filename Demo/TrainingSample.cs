using NeuralNetowrksAndDeepLearning;
using System.Linq;

namespace NeuralNetowrksAndDeepLearning.Demo
{
    public struct MNISTSample : ITrainingSample
    {
        public double[] Input { get; }

        public double[] Output { get; }

        public MNISTSample(byte[] input, byte output)
        {
            Input = input.Select(x => (double)x).ToArray();
            Output = Enumerable.Range(0, 10).Select(i => i == output ? 1.0 : 0.0).ToArray();
        }
    }
}