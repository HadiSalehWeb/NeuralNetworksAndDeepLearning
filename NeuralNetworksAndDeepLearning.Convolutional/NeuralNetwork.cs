using System;
using System.Collections.Generic;
using System.Linq;

//Todo: regularization

namespace NeuralNetworksAndDeepLearning.Convolutional
{
    public class NeuralNetwork
    {
        public int InputDimension { get; }
        public ILayer[] HiddenLayers { get; }
        public IOutputLayer OutputLayer { get; }
        public int Depth { get; }
        public ILayer[] Layers { get; }

        public NeuralNetwork(int inputDimension, IOutputLayer outputLayer, params ILayer[] hiddenLayers)
        {
            InputDimension = inputDimension;
            OutputLayer = outputLayer;
            HiddenLayers = hiddenLayers;
            Depth = 1 + hiddenLayers.Length;

            Layers = new ILayer[Depth];
            for (int i = 0; i < Depth; i++)
            {
                if (i == Depth - 1) Layers[i] = outputLayer;
                else Layers[i] = hiddenLayers[i];
                Layers[i].Initialize(i == 0 ? inputDimension : Layers[i - 1].WeightCount);
            }
        }

        public float Cost(IEnumerable<TrainingSample> data)
        {
            return data.Sum(d => OutputLayer.Cost(HiddenLayers.Aggregate(d.Input, (a, c) => c.Feedforward(a)), d.Output)) / data.Count();
        }

        public float[] Feedforward(float[] input)
        {
            return OutputLayer.Feedforward(HiddenLayers.Aggregate(input, (a, c) => c.Feedforward(a)));
        }

        public void SGD(TrainingSample[] trainingData, int epochs, int miniBatchSize, float learningRate/*, float regularizationRate = 0f*/, Action<int> onEpoch = null, Action<int> onBatch = null)
        {
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                var shuffledData = trainingData.Shuffle();
                for (int i = 0; i < trainingData.Length; i += miniBatchSize)
                {
                    RunMiniBatch(new ArraySegment<TrainingSample>(shuffledData, i, miniBatchSize), learningRate/*, regularizationRate*/);
                    onBatch?.Invoke(i / miniBatchSize);
                }
                onEpoch?.Invoke(epoch);
            }
        }

        private void RunMiniBatch(IEnumerable<TrainingSample> batch, float learningRate/*, float regularizationRate*/)
        {
            var costGradient = batch.AsParallel().Select(sample => Backpropagate(sample)).Aggregate((a, c) =>
            {
                for (int l = 0; l < a.Length; l++)
                    for (int i = 0; i < a[l].Length; i++)
                        a[l][i] += c[l][i];

                return a;
            });

            var factor = learningRate / batch.Count();

            for (int l = 0; l < costGradient.Length; l++)
                for (int i = 0; i < costGradient[l].Length; i++)
                    costGradient[l][i] *= factor;

            for (int i = 1; i < Depth; i++)
                Layers[i].UpdateWeights(costGradient[i]);
        }

        private float[][] Backpropagate(TrainingSample sample)
        {
            // Forwards pass
            var (weightedInputs, activations) = GetWeightedInputsAndActivations(sample.Input);

            var delCostOverDelWeightedInputs =
                OutputLayer.GetDelCostOverDelWeightedInputs(weightedInputs.Last(), activations.Last(), sample.Output);

            // Backwards pass
            float[][] delCostOverDelWeights = Layers.Select(l => new float[l.WeightCount]).ToArray();

            for (int l = Layers.Length - 1; l >= 0; l--)
            {
                delCostOverDelWeights[l] = Layers[l].BackpropagateWeights(delCostOverDelWeightedInputs);
                if (l != 0) delCostOverDelWeightedInputs = Layers[l].BackpropagateError(delCostOverDelWeightedInputs);

                ////Calculate ∂C/∂w for every w in the current layer:
                //for (int i = 0; i < Weights[l].GetLength(0); i++)
                //    for (int j = 0; j < Weights[l].GetLength(1); j++)
                //        delCostOverDelWeights[l][i, j] = // ∂C/∂w[l][i, j]
                //            delCostOverDelWeightedInputs[i] * // ∂C/∂z[l + 1][i]
                //            (j < Weights[l].GetLength(1) - 1 ? activations[l][j] : 1); // ∂z[l + 1][i]/∂w[l][i, j] = a[l][j] ((OR)) ∂z[l + 1][i]/∂b[l][i] = 1

                //// Calculate ∂C/∂a for the previous layer(a[l]):
                //if (l != 0)
                //{
                //    //var tempDelCostOverDelActivation = new float[Weights[l - 1].GetLength(0)];
                //    var tempDelCostOverDelWeightedInputs = new float[Weights[l - 1].GetLength(0)];
                //    for (int i = 0; i < Weights[l].GetLength(1) - 1; i++)
                //        for (int j = 0; j < Weights[l].GetLength(0); j++)
                //            tempDelCostOverDelWeightedInputs[i] += // ∂C/∂z[l][i] = sum over j:
                //                delCostOverDelWeightedInputs[j] * // ∂C/∂z[l + 1][j]
                //                Weights[l][j, i] * // ∂z[l + 1][j]/∂a[l][i] = w[l][j, i]
                //                MLMath.SigmoidPrime(weightedInputs[l][i]); // ∂a[l][i]/∂z[l][i] = σ′(z[l][i])
                //    delCostOverDelWeightedInputs = tempDelCostOverDelWeightedInputs;
                //}
            }

            return delCostOverDelWeights;
        }

        private (float[][], float[][]) GetWeightedInputsAndActivations(float[] input)
        {
            float[][] weightedInputs = new float[Depth - 1][], activations = new float[Depth - 1][];

            for (int i = 0; i < Depth - 1; i++)
            {
                weightedInputs[i] = Layers[i + 1].GetWeightedInput(input);
                activations[i] = Layers[i + 1].GetActivation(weightedInputs[i]);
                input = activations[i];
            }

            return (weightedInputs, activations);
        }
    }
}