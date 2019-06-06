using System.Linq;

namespace NeuralNetworksAndDeepLearning
{
    public class QuadraticCost : ICost
    {
        public double Cost(double[] activation, double[] outputs)
        {
            double cost = 0.0;

            for (int i = 0; i < activation.Length; i++)
            {
                double val = activation[i] - outputs[i];
                cost += val * val;
            }

            return .5 * cost;
        }

        public double Function(double activation, double output)
        {
            var val = activation - output;
            return .5 * val * val;
        }

        public double DelCostOverDelActivation(double activation, double output)
        {
            return activation - output;
        }

        public double DelCostOverDelWeightedInput(double weightedInput, double activation, double output)
        {
            return MLMath.SigmoidPrime(weightedInput) * DelCostOverDelActivation(activation, output);
        }
    }
}
