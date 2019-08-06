namespace NeuralNetworksAndDeepLearning.Interface
{
    public interface IHiddenLayer 
    {
        float[] Backprop(float[] delCostOverDelActivations, IForwardPropData ownForwardPropData, IForwardPropData nextForwardPropData);
        float[] BackpropagateDelCostOverDelActivations(float[] delCostOverDelActivations, IForwardPropData ownForwardPropData);
    }
}
