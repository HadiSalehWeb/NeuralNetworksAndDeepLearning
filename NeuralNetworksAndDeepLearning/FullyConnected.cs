using System;

//todo: formalize Random() creation?

namespace NeuralNetworksAndDeepLearning.Convolutional
{
    public abstract class FullyConnected : FullyConnectedBase, ILayer
    {
        protected readonly Func<float, float> activationFunction;
        protected readonly Func<float, float> activationDerivative;

        protected FullyConnected(int outputDimension, (Func<float, float>, Func<float, float>) activation) : base(outputDimension)
        {
            (activationFunction, activationDerivative) = activation;
        }

        public override float[] GetActivation(float[] weightedInput)
        {
            return MLMath.Activate(weightedInput, activationFunction);
        }
    }
}
