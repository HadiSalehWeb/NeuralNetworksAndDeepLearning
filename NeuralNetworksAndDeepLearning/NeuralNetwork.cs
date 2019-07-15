using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;

namespace NeuralNetworksAndDeepLearning
{
    public enum Regularization
    {
        None = 0,
        L1 = 1,
        L2 = 2
    }

    [Serializable]
    public class NeuralNetwork<TCost>
        where TCost : ICost, new()
    {
        public int LayerCount => Weights.Count + 1;
        public List<int> NeuronCount => new List<int> { Weights[0].GetLength(1) - 1 }.Concat(Weights.Select(x => x.GetLength(0))).ToList();
        public List<float[,]> Weights { get; }
        protected readonly TCost cost;
        protected readonly Regularization regularization;

        public NeuralNetwork(List<int> layers, Regularization regularization, TCost cost, Random rand)
        {
            if (layers == null) throw new ArgumentNullException(nameof(layers));
            if (rand == null) throw new ArgumentNullException(nameof(rand));
            if (layers.Count <= 1) throw new ArgumentException("The network must contain at least two layers.", nameof(layers));
            if (layers.Any(l => l <= 0)) throw new ArgumentException("Each layer of the network must contain at least one neuron.", nameof(layers));

            this.regularization = regularization;
            this.cost = cost;
            Weights = new List<float[,]>();

            for (int i = 0; i < layers.Count - 1; i++)
            {
                Weights.Add(new float[layers[i + 1], layers[i] + 1]);
                var standardDeviation = (float)(1f / Math.Sqrt(layers[i]));
                for (int x = 0; x < layers[i + 1]; x++)
                    for (int y = 0; y < layers[i] + 1; y++)
                        Weights[i][x, y] = MLMath.Gaussian(rand, 0f, standardDeviation);
            }
        }
        public NeuralNetwork(List<int> layers, Regularization regularization, TCost cost) : this(layers, regularization, cost, new Random()) { }
        public NeuralNetwork(List<int> layers, Regularization regularization) : this(layers, regularization, new TCost()) { }
        public NeuralNetwork(List<int> layers) : this(layers, Regularization.None) { }
        public NeuralNetwork() : this(new List<int> { 1, 1 }) { }

        public NeuralNetwork(List<float[,]> weights, Regularization regularization, TCost cost)
        {
            if (weights == null || weights.Any(l => l == null)) throw new ArgumentNullException(nameof(weights));

            if (weights.Count <= 0) throw new ArgumentException("The network must contain at least two layers.", nameof(weights));
            if (weights[0].GetLength(1) <= 1 || weights.Any(l => l.GetLength(0) < 1)) throw new ArgumentException("Each layer of the network must contain at least one neuron.", nameof(weights));

            this.regularization = regularization;
            this.cost = cost;
            Weights = weights;
        }
        public NeuralNetwork(List<float[,]> weights, Regularization regularization) : this(weights, regularization, new TCost()) { }
        public NeuralNetwork(List<float[,]> weights) : this(weights, Regularization.None, new TCost()) { }

        public float[] Feedforward(float[] vector)
        {
            if (vector.Length != Weights.First().GetLength(1) - 1) throw new ArgumentException("Mismatching dimensions.");

            foreach (var weightsLayer in Weights)
            {
                float[] res = new float[weightsLayer.GetLength(0)];
                for (int i = 0; i < weightsLayer.GetLength(0); i++)
                {
                    float weightedInput = 0;
                    for (int j = 0; j < weightsLayer.GetLength(1) - 1; j++)
                        weightedInput += weightsLayer[i, j] * vector[j];
                    res[i] = MLMath.Sigmoid(weightedInput + weightsLayer[i, weightsLayer.GetLength(1) - 1]);
                }
                vector = res;
            }

            return vector;
        }

        public int FeedforwardMaxArg(float[] vector)
        {
            var output = Feedforward(vector);
            var max = float.MinValue;
            var maxIndex = -1;
            for (int i = 0; i < output.Length; i++)
                if (output[i] > max)
                    (max, maxIndex) = (output[i], i);
            return maxIndex;
        }

        public float Cost(IEnumerable<TrainingSample> trainingData)
        {
            float costValue = 0;

            foreach (var sample in trainingData)
            {
                float[] activations = Feedforward(sample.Input);
                for (int i = 0; i < activations.Length; i++)
                    costValue += cost.Function(activations[i], sample.Output[i]);
            }

            return costValue / trainingData.Count();
        }

        public int Validate(IEnumerable<TrainingSample> trainingData, Func<float[], float[], bool> ValidateSample)
        {
            return trainingData.Aggregate(0, (a, c) => ValidateSample(Feedforward(c.Input), c.Output) ? a + 1 : a);
        }

        public void SGD(IEnumerable<TrainingSample> trainingData, int epochs, int miniBatchSize, float learningRate, float regularizationRate = 0f, Action<int> onEpoch = null, Action<int> onBatch = null)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                var shuffledData = trainingData.Shuffle();

                for (int i = 0; i < shuffledData.Count; i += miniBatchSize)
                {
                    UpdateMiniBatch(shuffledData.Skip(i).Take(miniBatchSize), learningRate, shuffledData.Count, regularizationRate);
                    onBatch?.Invoke(i);
                }

                onEpoch?.Invoke(epoch);
            }
        }

        protected void UpdateMiniBatch(IEnumerable<TrainingSample> batch, float learningRate, int trainingDataCount, float regularizationRate)
        {
            var costGradient = batch.AsParallel().Select(sample => Backpropagate(sample)).Aggregate((a, c) =>
            {
                for (int l = 0; l < a.Count; l++)
                    for (int i = 0; i < a[l].GetLength(0); i++)
                        for (int j = 0; j < a[l].GetLength(1); j++)
                            a[l][i, j] += c[l][i, j];

                return a;
            });

            float factor = learningRate / batch.Count(), regularizationFactor = learningRate * regularizationRate / trainingDataCount;

            switch (regularization)
            {
                case Regularization.L1:
                    for (int l = 0; l < costGradient.Count; l++)
                        for (int i = 0; i < Weights[l].GetLength(0); i++)
                            for (int j = 0; j < Weights[l].GetLength(1); j++)
                                Weights[l][i, j] =
                                    Weights[l][i, j]
                                    - (j == Weights[l].GetLength(1) - 1 ? 1 : regularizationFactor * Math.Sign(Weights[l][i, j]))
                                    - factor * costGradient[l][i, j];

                    break;
                case Regularization.L2:
                    for (int l = 0; l < costGradient.Count; l++)
                        for (int i = 0; i < Weights[l].GetLength(0); i++)
                            for (int j = 0; j < Weights[l].GetLength(1); j++)
                                Weights[l][i, j] =
                                    (j == Weights[l].GetLength(1) - 1 ? 1 : 1 - regularizationFactor) * Weights[l][i, j]
                                    - factor * costGradient[l][i, j];

                    break;
                case Regularization.None:
                default:
                    for (int l = 0; l < costGradient.Count; l++)
                        for (int i = 0; i < Weights[l].GetLength(0); i++)
                            for (int j = 0; j < Weights[l].GetLength(1); j++)
                                Weights[l][i, j] = Weights[l][i, j] - factor * costGradient[l][i, j];
                    break;
            }
        }

        /// <summary>
        /// Returns the partial derivative of the cost function on one sample with respect to every weight in the network.
        /// </summary>
        public List<float[,]> Backpropagate(TrainingSample sample)
        {
            // Forwards pass
            var (weightedInputs, activations) = GetWeightedInputsAndActivations(sample.Input);

            var delCostOverDelWeightedInputs = activations.Last().Select((a, i) => cost.DelCostOverDelWeightedInput(weightedInputs.Last()[i], a, sample.Output[i])).ToArray();

            // Backwards pass
            List<float[,]> delCostOverDelWeights = Weights.Select(x => new float[x.GetLength(0), x.GetLength(1)]).ToList();

            for (int l = Weights.Count - 1; l >= 0; l--)
            {
                //Calculate ∂C/∂w for every w in the current layer:
                for (int i = 0; i < Weights[l].GetLength(0); i++)
                    for (int j = 0; j < Weights[l].GetLength(1); j++)
                        delCostOverDelWeights[l][i, j] = // ∂C/∂w[l][i, j]
                            delCostOverDelWeightedInputs[i] * // ∂C/∂z[l + 1][i]
                            (j < Weights[l].GetLength(1) - 1 ? activations[l][j] : 1); // ∂z[l + 1][i]/∂w[l][i, j] = a[l][j] ((OR)) ∂z[l + 1][i]/∂b[l][i] = 1

                // Calculate ∂C/∂a for the previous layer(a[l]):
                if (l != 0)
                {
                    //var tempDelCostOverDelActivation = new float[Weights[l - 1].GetLength(0)];
                    var tempDelCostOverDelWeightedInputs = new float[Weights[l - 1].GetLength(0)];
                    for (int i = 0; i < Weights[l].GetLength(1) - 1; i++)
                        for (int j = 0; j < Weights[l].GetLength(0); j++)
                            tempDelCostOverDelWeightedInputs[i] += // ∂C/∂z[l][i] = sum over j:
                                delCostOverDelWeightedInputs[j] * // ∂C/∂z[l + 1][j]
                                Weights[l][j, i] * // ∂z[l + 1][j]/∂a[l][i] = w[l][j, i]
                                MLMath.SigmoidPrime(weightedInputs[l][i]); // ∂a[l][i]/∂z[l][i] = σ′(z[l][i])
                    delCostOverDelWeightedInputs = tempDelCostOverDelWeightedInputs;
                }
            }

            return delCostOverDelWeights;
        }

        public (List<float[]>, List<float[]>) GetWeightedInputsAndActivations(float[] input)
        {
            List<float[]> activations = new List<float[]>() { input }.Concat(Weights.Select(x => new float[x.GetLength(0)])).ToList();
            List<float[]> weightedInputs = activations.Select(x => new float[x.Length]).ToList();

            for (int l = 0; l < Weights.Count; l++)
                for (int i = 0; i < Weights[l].GetLength(0); i++)
                {
                    float value = 0;
                    for (int j = 0; j < Weights[l].GetLength(1) - 1; j++)
                        value += Weights[l][i, j] * activations[l][j];// weights
                    weightedInputs[l + 1][i] = value + Weights[l][i, Weights[l].GetLength(1) - 1];// bias
                    activations[l + 1][i] = MLMath.Sigmoid(weightedInputs[l + 1][i]);
                }

            return (weightedInputs, activations);
        }

        public void Save(string path)
        {
            using (var stream = File.OpenWrite(path))
                new BinaryFormatter().Serialize(stream, Weights);
        }

        public static NeuralNetwork<TCost> Load(string path)
        {
            List<float[,]> weights;
            using (var stream = File.OpenRead(path))
                weights = (List<float[,]>)new BinaryFormatter().Deserialize(stream);

            return new NeuralNetwork<TCost>(weights);
        }

        public override string ToString()
        {
            StringBuilder builder = new StringBuilder();
            builder.AppendLine($"Shape: { string.Join(", ", NeuronCount) }");
            builder.AppendLine($"Weights:");
            int layerIndex = 0;

            foreach (var layer in Weights)
            {
                builder.AppendLine($"Layer #{ layerIndex++ }:");
                builder.AppendLine();
                for (int i = 0; i < layer.GetLength(0); i++)
                {
                    for (int j = 0; j < layer.GetLength(1); j++)
                    {
                        builder.Append(layer[i, j]);
                        if (j != layer.GetLength(1) - 1)
                            builder.Append(", ");
                    }
                    if (i != layer.GetLength(0) - 1)
                        builder.AppendLine();
                }
            }

            return builder.ToString();
        }
    }
}
