using NeuralNetworksAndDeepLearning.Interface;

namespace NeuralNetworksAndDeepLearning.Model
{
    public class ConvolutionalForwardPropData : IForwardPropData
    {
        public float[] WeightedInputs { get; }
        public float[] Activations { get; }
        public object MaxPooling { get; }

        public ConvolutionalForwardPropData(float[] weightedInputs, float[] activations, object maxPooling)//todo formalize this
        {
            WeightedInputs = weightedInputs;
            Activations = activations;
            MaxPooling = maxPooling;
        }
    }
}
