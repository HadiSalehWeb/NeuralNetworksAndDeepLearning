namespace NeuralNetworksAndDeepLearning.Interface
{
    public interface ILayer
    {
        int ParameterCount { get; }
        int OutputDimension { get; }

        void Initialize(int outputDimensionOfPreviousLayer);

        float[] GetWeightedInput(float[] input);
        float[] GetActivation(float[] weightedInput);
        float[] Feedforward(float[] input);

        IForwardPropData ForwardProp(float[] previousActivations);

        void UpdateParameters(float[] costGradient);
    }
}
