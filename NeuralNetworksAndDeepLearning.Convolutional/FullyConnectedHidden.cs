using System;

namespace NeuralNetworksAndDeepLearning.Convolutional
{
    public class FullyConnectedHidden : FullyConnected, IHiddenLayer
    {
        public FullyConnectedHidden(int outputDimension, (Func<float, float>, Func<float, float>) activation) : base(outputDimension, activation) { }

        public float[] BackpropagateParameters(float[] delCostOverDelActivations, float[] outWeightedInputs, float[] inActivations)
        {
            float[] gradient = new float[WeightMatrix.GetLength(0) * WeightMatrix.GetLength(1)];
            for (int i = 0; i < OutputDimension; i++)
            {
                var index = i * (InputDimension + 1);
                var delCostOverDelBias = delCostOverDelActivations[i] * activationDerivative(outWeightedInputs[i]);

                for (int j = 0; j < InputDimension; j++)
                    gradient[index + j] = delCostOverDelBias * inActivations[j];

                gradient[index + InputDimension] = delCostOverDelBias;
            }

            return gradient;
        }

        public float[] BackpropagateDelCostOverDelActivations(float[] delCostOverDelActivations, float[] outWeightedInputs)
        {
            float[] del = new float[InputDimension];

            for (int i = 0; i < InputDimension; i++)
            {
                del[i] = 0f;

                for (int j = 0; j < OutputDimension; j++)
                    del[i] += delCostOverDelActivations[j] * activationDerivative(outWeightedInputs[j]) * WeightMatrix[j, i];
            }

            return del;
        }
    }
}
