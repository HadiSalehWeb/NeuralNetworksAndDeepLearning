using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace NeuralNetworksAndDeepLearning.Visualizer
{
    public class FeatureExtractor<TCost>
        where TCost : ICost, new()
    {
        public NeuralNetwork<TCost> Network { get; }
        private readonly TCost cost;
        public FeatureExtractor(NeuralNetwork<TCost> network)
        {
            Network = network;
            cost = new TCost();
        }

        public float[] Enhance(float[] input, float[] desiredOutput)
        {
            return Util.Add(input, BackpropagateInputLayer(input, desiredOutput));
        }

        private float[] BackpropagateInputLayer(float[] input, float[] desiredOutput)
        {
            // Forwards pass
            var (weightedInputs, activations) = Network.GetWeightedInputsAndActivations(input);

            var delCostOverDelWeightedInputs = activations.Last().Select((a, i) => cost.DelCostOverDelWeightedInput(weightedInputs.Last()[i], a, desiredOutput[i])).ToArray();

            var Weights = Network.Weights;

            // Backwards pass
            for (int l = Weights.Count - 1; l >= 0; l--)
            {
                // Calculate ∂C/∂a for the previous layer(a[l]):
                //var tempDelCostOverDelActivation = new float[Weights[l - 1].GetLength(0)];
                var tempDelCostOverDelWeightedInputs = new float[Weights[l].GetLength(1) - 1];
                for (int i = 0; i < Weights[l].GetLength(1) - 1; i++)
                    for (int j = 0; j < Weights[l].GetLength(0); j++)
                        tempDelCostOverDelWeightedInputs[i] += // ∂C/∂z[l][i] = sum over j:
                            delCostOverDelWeightedInputs[j] * // ∂C/∂z[l + 1][j]
                            Weights[l][j, i] * // ∂z[l + 1][j]/∂a[l][i] = w[l][j, i]
                            (l == 0 ? 1 : MLMath.SigmoidPrime(weightedInputs[l][i])); // ∂a[l][i]/∂z[l][i] = σ′(z[l][i])
                delCostOverDelWeightedInputs = tempDelCostOverDelWeightedInputs;
            }

            return delCostOverDelWeightedInputs;
        }
    }
}
