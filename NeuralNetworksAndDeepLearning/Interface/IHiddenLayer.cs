namespace NeuralNetworksAndDeepLearning.Convolutional
{
    public interface IHiddenLayer : ILayer
    {
        float[] BackpropagateParameters(float[] delCostOverDelActivations, float[] outWeightedInputs, float[] inActivations);
        float[] BackpropagateDelCostOverDelActivations(float[] delCostOverDelActivations, float[] outWeightedInputs);
    }
}
