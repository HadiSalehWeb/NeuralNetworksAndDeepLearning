using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;

namespace NeuralNetworksAndDeepLearning
{
    [Serializable]
    public class NeuralNetwork
    {
        public int LayerCount => Weights.Count + 1;
        public List<int> NeuronCount => new List<int> { Weights[0].GetLength(1) - 1 }.Concat(Weights.Select(x => x.GetLength(0))).ToList();
        public List<double[,]> Weights { get; }

        public NeuralNetwork(List<int> layers, Random rand)
        {
            if (layers == null) throw new ArgumentNullException(nameof(layers));
            if (rand == null) throw new ArgumentNullException(nameof(rand));
            if (layers.Count <= 1) throw new ArgumentException("The network must contain at least two layers.", nameof(layers));
            if (layers.Any(l => l <= 0)) throw new ArgumentException("Each layer of the network must contain at least one neuron.", nameof(layers));

            Weights = new List<double[,]>();
            for (int i = 0; i < layers.Count - 1; i++)
            {
                Weights.Add(new double[layers[i + 1], layers[i] + 1]);
                for (int x = 0; x < layers[i + 1]; x++)
                    for (int y = 0; y < layers[i] + 1; y++)
                        Weights[i][x, y] = rand.NextDouble() * 3.0 - 1.5;
            }
        }

        public NeuralNetwork(List<int> layers) : this(layers, new Random()) { }

        public NeuralNetwork(List<double[,]> weights)
        {
            if (weights == null || weights.Any(l => l == null)) throw new ArgumentNullException(nameof(weights));

            if (weights.Count <= 0) throw new ArgumentException("The network must contain at least two layers.", nameof(weights));
            if (weights[0].GetLength(1) <= 1 || weights.Any(l => l.GetLength(0) < 1)) throw new ArgumentException("Each layer of the network must contain at least one neuron.", nameof(weights));

            Weights = weights;
        }

        public double[] Feedforward(double[] vector)
        {
            if (vector.Length != Weights.First().GetLength(1) - 1) throw new ArgumentException("Mismatching dimensions.");

            foreach (var weightsLayer in Weights)
            {
                double[] res = new double[weightsLayer.GetLength(0)];
                for (int i = 0; i < weightsLayer.GetLength(0); i++)
                {
                    double weightedInput = 0;
                    for (int j = 0; j < weightsLayer.GetLength(1) - 1; j++)
                        weightedInput += weightsLayer[i, j] * vector[j];
                    res[i] = Sigmoid(weightedInput + weightsLayer[i, weightsLayer.GetLength(1) - 1]);
                }
                vector = res;
            }

            return vector;
        }

        public int FeedforwardMaxArg(double[] vector)
        {
            var output = Feedforward(vector);
            var max = double.MinValue;
            var maxIndex = -1;
            for (int i = 0; i < output.Length; i++)
                if (output[i] > max)
                    (max, maxIndex) = (output[i], i);
            return maxIndex;
        }

        public double Cost(IEnumerable<TrainingSample> trainingData)
        {
            double cost = 0;

            foreach (var sample in trainingData)
            {
                double[] result = Feedforward(sample.Input);
                for (int i = 0; i < result.Length; i++)
                {
                    result[i] = sample.Output[i] - result[i];
                    cost += result[i] * result[i];
                }
            }

            return cost / (2 * trainingData.Count());
        }

        public int Validate(IEnumerable<TrainingSample> trainingData, Func<double[], double[], bool> ValidateSample)
        {
            return trainingData.Aggregate(0, (a, c) => a + (ValidateSample(Feedforward(c.Input), c.Output) ? 1 : 0));
        }

        public void SGD(IEnumerable<TrainingSample> trainingData, int epochs, int miniBatchSize, double learningRate, Action<int> onEpoch = null, Action<int> onBatch = null)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                var shuffledData = Shuffle(trainingData);

                for (int i = 0; i < shuffledData.Count; i += miniBatchSize)
                {
                    UpdateMiniBatch(shuffledData.Skip(i).Take(miniBatchSize), learningRate);
                    onBatch?.Invoke(i);
                }

                onEpoch?.Invoke(epoch);
            }
        }

        public void UpdateMiniBatch(IEnumerable<TrainingSample> batch, double learningRate)
        {
            var costGradient = batch.AsParallel().Select(sample => Backpropagate(sample)).Aggregate((a, c) =>
            {
                for (int l = 0; l < a.Count; l++)
                    for (int i = 0; i < a[l].GetLength(0); i++)
                        for (int j = 0; j < a[l].GetLength(1); j++)
                            a[l][i, j] += c[l][i, j];

                return a;
            });

            var factor = learningRate / batch.Count();

            for (int l = 0; l < costGradient.Count; l++)
                for (int i = 0; i < Weights[l].GetLength(0); i++)
                    for (int j = 0; j < Weights[l].GetLength(1); j++)
                        Weights[l][i, j] -= factor * costGradient[l][i, j];
        }

        /// <summary>
        /// Returns the partial derivative of the cost function on one sample with respect to every weight in the network.
        /// </summary>
        public List<double[,]> Backpropagate(TrainingSample sample)
        {
            // Forwards pass
            var (weightedInputs, activations) = GetWeightedInputsAndActivations(sample.Input);

            var delCostOverDelActivation = activations.Last().Select((a, i) => a - sample.Output[i]).ToArray();

            // Backwards pass
            List<double[,]> delCostOverDelWeights = Weights.Select(x => new double[x.GetLength(0), x.GetLength(1)]).ToList();

            for (int l = Weights.Count - 1; l >= 0; l--)
            {
                //Calculate ∂C/∂w for every w in the current layer:
                for (int i = 0; i < Weights[l].GetLength(0); i++)
                {
                    var delCostOverDelWeightedInput = // ∂C/∂z[l + 1][i]
                        delCostOverDelActivation[i] * // ∂C/∂a[l + 1][i]
                        SigmoidPrime(weightedInputs[l + 1][i]); // ∂a[l + 1][i]/∂z[l + 1][i] = ∂(σ(z[l + 1][i]))/∂z[l + 1][i] = σ′(z[l + 1][i])

                    for (int j = 0; j < Weights[l].GetLength(1); j++)
                        delCostOverDelWeights[l][i, j] = // ∂C/∂w[l][i, j]
                            delCostOverDelWeightedInput * // ∂C/∂z[l + 1][i]
                            (j < Weights[l].GetLength(1) - 1 ? activations[l][j] : 1); // ∂z[l + 1][i]/∂w[l][i, j] = a[l][j] ((OR)) ∂z[l + 1][i]/∂b[l][i] = 1
                }

                // Calculate ∂C/∂a for the previous layer(a[l]):
                if (l != 0)
                {
                    var tempDelCostOverDelActivation = new double[Weights[l - 1].GetLength(0)];
                    for (int i = 0; i < Weights[l - 1].GetLength(0); i++)
                        for (int j = 0; j < Weights[l].GetLength(0); j++)
                            tempDelCostOverDelActivation[i] += // ∂C/∂a[l][i] = sum over j:
                                delCostOverDelActivation[j] * // ∂C/∂a[l + 1][j]
                                SigmoidPrime(weightedInputs[l + 1][j]) * // ∂a[l + 1][j]/∂z[l + 1][j] = ∂(σ(z[l + 1][j]))/∂z[l + 1][j] = σ′(z[l + 1][j])
                                Weights[l][j, i]; // ∂z[l + 1][j]/∂a[l][i] = w[l][j, i]
                    delCostOverDelActivation = tempDelCostOverDelActivation;
                }
            }

            return delCostOverDelWeights;
        }

        public (List<double[]>, List<double[]>) GetWeightedInputsAndActivations(double[] input)
        {
            List<double[]> activations = new List<double[]>() { input }.Concat(Weights.Select(x => new double[x.GetLength(0)])).ToList();
            List<double[]> weightedInputs = activations.Select(x => new double[x.Length]).ToList();

            for (int l = 0; l < Weights.Count; l++)
                for (int i = 0; i < Weights[l].GetLength(0); i++)
                {
                    double value = 0;
                    for (int j = 0; j < Weights[l].GetLength(1) - 1; j++)
                        value += Weights[l][i, j] * activations[l][j];// weights
                    weightedInputs[l + 1][i] = value + Weights[l][i, Weights[l].GetLength(1) - 1];// bias
                    activations[l + 1][i] = Sigmoid(weightedInputs[l + 1][i]);
                }

            return (weightedInputs, activations);
        }

        public void Save(string path)
        {
            using (var stream = File.OpenWrite(path))
                new BinaryFormatter().Serialize(stream, this);
        }

        public static NeuralNetwork Load(string path)
        {
            using (var stream = File.OpenRead(path))
                return (NeuralNetwork)new BinaryFormatter().Deserialize(stream);
        }

        public override string ToString()
        {
            StringBuilder builder = new StringBuilder();
            builder.AppendLine($"Shape: { String.Join(", ", NeuronCount) }");
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

        private static double Sigmoid(double z)
        {
            return 1 / (1 + Math.Exp(-z));
        }

        private static double SigmoidPrime(double z)
        {
            return Sigmoid(z) * (1 - Sigmoid(z));
        }

        private static List<T> Shuffle<T>(IEnumerable<T> data, Random rand)
        {
            List<T> ts = new List<T>(data);
            List<T> result = new List<T>();

            while (ts.Any())
            {
                int r = rand.Next(0, ts.Count);
                result.Add(ts[r]);
                ts.RemoveAt(r);
            }

            return result;
        }

        private static List<T> Shuffle<T>(IEnumerable<T> data)
        {
            return Shuffle(data, new Random());
        }
    }
}
