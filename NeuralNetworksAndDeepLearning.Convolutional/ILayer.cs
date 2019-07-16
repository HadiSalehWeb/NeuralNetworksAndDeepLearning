namespace NeuralNetworksAndDeepLearning.Convolutional
{
    public interface ILayer
    {
        int ParameterCount { get; }
        int OutputDimension { get; }

        void Initialize(int outputDimensionOfPreviousLayer);
        float[] GetWeightedInput(float[] input);
        float[] GetActivation(float[] weightedInput);
        float[] Feedforward(float[] input);
        void UpdateParameters(float[] costGradient);
    }
}
