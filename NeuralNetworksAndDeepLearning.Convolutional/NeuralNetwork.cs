using System;
using System.Collections.Generic;
using System.Linq;

//Todo: regularization

namespace NeuralNetworksAndDeepLearning.Convolutional
{
    public class NeuralNetwork
    {
        public int InputDimension { get; }
        public IHiddenLayer[] HiddenLayers { get; }
        public IOutputLayer OutputLayer { get; }
        public int Depth { get; }
        public ILayer[] Layers { get; }

        public NeuralNetwork(int inputDimension, IOutputLayer outputLayer, params IHiddenLayer[] hiddenLayers)
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
                Layers[i].Initialize(i == 0 ? inputDimension : Layers[i - 1].OutputDimension);
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
                Layers[i].UpdateParameters(costGradient[i]);
        }

        private float[][] Backpropagate(TrainingSample sample)
        {
            // Forwards pass
            var (weightedInputs, activations) = GetWeightedInputsAndActivations(sample.Input);

            // Backwards pass
            float[][] costGradient = Layers.Select(l => new float[l.ParameterCount]).ToArray();

            var outputError =
                OutputLayer.GetError(weightedInputs.Last(), activations.Last(), sample.Output);

            costGradient[Depth - 1] = OutputLayer.BackpropagateParameters(outputError);

            var delCostOverDelActivations = OutputLayer.BackpropagateErrorToActivation(outputError);

            for (int l = HiddenLayers.Length - 1; l >= 0; l--)
            {
                costGradient[l] = HiddenLayers[l].BackpropagateParameters(delCostOverDelActivations, weightedInputs[l + 1], activations[l]);
                if (l != 0) delCostOverDelActivations = HiddenLayers[l].BackpropagateDelCostOverDelActivations(delCostOverDelActivations, weightedInputs[l + 1]);
            }

            return costGradient;
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