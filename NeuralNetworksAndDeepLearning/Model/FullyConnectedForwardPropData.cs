using NeuralNetworksAndDeepLearning.Interface;

namespace NeuralNetworksAndDeepLearning.Model
{
    public class FullyConnectedForwardPropData : IForwardPropData
    {
        public float[] WeightedInputs { get; }
        public float[] Activations { get; }

        public FullyConnectedForwardPropData(float[] weightedInputs, float[] activations)
        {
            WeightedInputs = weightedInputs;
            Activations = activations;
        }
    }
}
