using System;

//todo: formalize Random() creation?

namespace NeuralNetworksAndDeepLearning.Layer
{
    public abstract class FullyConnected : FullyConnectedBase
    {
        protected readonly Func<float, float> activationFunction;
        protected readonly Func<float, float> activationDerivative;

        protected FullyConnected(int outputDimension, (Func<float, float>, Func<float, float>) activation) : base(outputDimension)
        {
            (activationFunction, activationDerivative) = activation;
        }

        public override float[] GetActivation(float[] weightedInput)
        {
            float[] activation = new float[weightedInput.Length];

            for (int i = 0; i < weightedInput.Length; i++)
                activation[i] = activationFunction(weightedInput[i]);

            return activation;
        }
    }
}
