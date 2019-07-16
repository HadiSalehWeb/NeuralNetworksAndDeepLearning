using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworksAndDeepLearning.Convolutional
{
    public class FullyConnectedHidden : FullyConnected, IHiddenLayer
    {
        public FullyConnectedHidden(int outputDimension, ActivationFunctionType activation) : base(outputDimension, activation) { }

        public float[] BackpropagateParameters(float[] delCostOverDelActivations, float[] outWeightedInputs, float[] inActivations)
        {
            float[] ret = new float[WeightMatrix.GetLength(0) * WeightMatrix.GetLength(1)];
            for (int i = 0; i < OutputDimension; i++)
            {
                var index = i * (InputDimension + 1);
                var delCostOverDelBias = delCostOverDelActivations[i] * activationDerivative(outWeightedInputs[i]);

                for (int j = 0; j < InputDimension; j++)
                    ret[index + j] = delCostOverDelBias * inActivations[j];

                ret[index + InputDimension] = delCostOverDelBias;
            }

            return ret;
        }

        public float[] BackpropagateDelCostOverDelActivations(float[] delCostOverDelActivations, float[] outWeightedInputs)
        {
            float[] ret = new float[InputDimension];

            for (int i = 0; i < InputDimension; i++)
            {
                ret[i] = 0f;

                for (int j = 0; j < OutputDimension; j++)
                    ret[i] += delCostOverDelActivations[j] * activationDerivative(outWeightedInputs[j]) * WeightMatrix[j, i];
            }

            return ret;
        }
    }
}
