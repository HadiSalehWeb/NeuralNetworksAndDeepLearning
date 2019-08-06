namespace NeuralNetworksAndDeepLearning.Interface
{
    public interface IForwardPropData
    {
        float[] WeightedInputs { get; }
        float[] Activations { get; }
    }
}
