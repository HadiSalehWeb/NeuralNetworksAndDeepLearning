namespace NeuralNetworksAndDeepLearning.Interface
{
    public interface IOutputLayer
    {
        float Cost(float[] previousActivations, float[] output);
        float[] GetError(float[] output, IForwardPropData ownForwardPropData);
        float[] Backprop(float[] error, IForwardPropData ownForwardPropData);
        float[] BackpropagateErrorToActivation(float[] error);
    }
}
