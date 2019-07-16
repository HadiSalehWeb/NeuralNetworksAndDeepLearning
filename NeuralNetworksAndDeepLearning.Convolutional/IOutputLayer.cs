namespace NeuralNetworksAndDeepLearning.Convolutional
{
    public interface IOutputLayer : ILayer
    {
        float Cost(float[] previousActivations, float[] output);
        float[] GetError(float[] output, float[] activations, float[] weightedInputs);
        float[] BackpropagateParameters(float[] error);
        float[] BackpropagateErrorToActivation(float[] error);
    }
}
