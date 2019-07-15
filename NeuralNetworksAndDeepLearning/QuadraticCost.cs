using System.Linq;

namespace NeuralNetworksAndDeepLearning
{
    public class QuadraticCost : ICost
    {
        public float Cost(float[] activation, float[] outputs)
        {
            float cost = 0f;

            for (int i = 0; i < activation.Length; i++)
            {
                float val = activation[i] - outputs[i];
                cost += val * val;
            }

            return .5f * cost;
        }

        public float Function(float activation, float output)
        {
            var val = activation - output;
            return .5f * val * val;
        }

        public float DelCostOverDelActivation(float activation, float output)
        {
            return activation - output;
        }

        public float DelCostOverDelWeightedInput(float weightedInput, float activation, float output)
        {
            return MLMath.SigmoidPrime(weightedInput) * DelCostOverDelActivation(activation, output);
        }
    }
}
