using System;
using NeuralNetworksAndDeepLearning.Interface;
using NeuralNetworksAndDeepLearning.Model;

namespace NeuralNetworksAndDeepLearning.Layer
{
    public class FullyConnectedHidden : FullyConnected, IHiddenLayer
    {
        public FullyConnectedHidden(int outputDimension, (Func<float, float>, Func<float, float>) activation) : base(outputDimension, activation) { }

        public float[] Backprop(float[] delCostOverDelActivations, IForwardPropData ownForwardPropData, IForwardPropData nextForwardPropData)
        {
            float[] gradient = new float[WeightMatrix.GetLength(0) * WeightMatrix.GetLength(1)];
            for (int i = 0; i < OutputDimension; i++)
            {
                var index = i * (InputDimension + 1);
                var delCostOverDelBias = delCostOverDelActivations[i] * activationDerivative(ownForwardPropData.WeightedInputs[i]);

                for (int j = 0; j < InputDimension; j++)
                    gradient[index + j] = delCostOverDelBias * nextForwardPropData.Activations[j];

                gradient[index + InputDimension] = delCostOverDelBias;
            }

            return gradient;
        }

        public float[] BackpropagateDelCostOverDelActivations(float[] delCostOverDelActivations, IForwardPropData ownForwardPropData)
        {
            float[] del = new float[InputDimension];

            for (int i = 0; i < InputDimension; i++)
            {
                del[i] = 0f;

                for (int j = 0; j < OutputDimension; j++)
                    del[i] += delCostOverDelActivations[j] * activationDerivative(ownForwardPropData.WeightedInputs[j]) * WeightMatrix[j, i];
            }

            return del;
        }
    }
}
