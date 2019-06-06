using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NeuralNetworksAndDeepLearning.Visualizer;

namespace NeuralNetworksAndDeepLearning.Demo
{
    class Program
    {
        const int BATCH_SIZE = 10;
        const int EPOCH_COUNT = 30;
        const string TRAINING_IMAGES_PATH = @"C:\Users\hadis\source\repos\NeuralNetworksAndDeepLearning\Demo\Training Data\train-images.idx3-ubyte";
        const string TRAINING_LABELS_PATH = @"C:\Users\hadis\source\repos\NeuralNetworksAndDeepLearning\Demo\Training Data\train-labels.idx1-ubyte";
        const string TEST_IMAGES_PATH = @"C:\Users\hadis\source\repos\NeuralNetworksAndDeepLearning\Demo\Training Data\t10k-images.idx3-ubyte";
        const string TEST_LABELS_PATH = @"C:\Users\hadis\source\repos\NeuralNetworksAndDeepLearning\Demo\Training Data\t10k-labels.idx1-ubyte";
        const int INPUT_COUNT = 784;
        const int HIDDEN_COUNT = 30;
        const int OUTPUT_COUNT = 10;
        const double LEARNING_RATE = 0.5;
        const double REGULARIZATION_RATE = 5.0;

        static void Main(string[] args)
        {
            //TrainOnMnistAndSave();
            VisualizeNetwork(
                @"C:\Users\hadis\source\repos\NeuralNetworksAndDeepLearning\Demo\nets",
                @"C:\Users\hadis\source\repos\NeuralNetworksAndDeepLearning\Demo\Visualizations",
                @"cross-entropy"
            );
        }

        public static void TrainOnMnistAndSave(bool printBatchProgress = false, bool printCostEveryEpoch = false, bool printFetchStats = false)
        {
            var network = new NeuralNetwork<CrossEntropy>(new List<int>() { INPUT_COUNT, HIDDEN_COUNT, OUTPUT_COUNT }, Regularization.L2);

            var rawData = FetchData(TRAINING_LABELS_PATH, TRAINING_IMAGES_PATH, printFetchStats);
            var trainingData = rawData.Take(50000).ToArray();
            var validationData = rawData.Skip(50000).ToArray();
            var testData = FetchData(TEST_LABELS_PATH, TEST_IMAGES_PATH, printFetchStats);
            Action<int> onBatch = null;

            if (printBatchProgress)
            {
                var batchCount = trainingData.Length / BATCH_SIZE;
                onBatch = i =>
                {
                    Console.WriteLine($"Batch progress: { i }/{ batchCount }");
                    Console.SetCursorPosition(0, Console.CursorTop - 1);
                };
            }
            Console.WriteLine($"Starting Cost: { (printCostEveryEpoch ? network.Cost(testData) : -1) }, Accuracy: { network.Validate(testData, (a, o) => ValidateSample(a, o)) } / { testData.Count() }");

            network.SGD(trainingData, EPOCH_COUNT, BATCH_SIZE, LEARNING_RATE, REGULARIZATION_RATE, i =>
            {
                Console.WriteLine($"Finished epoch { i }. Cost: { (printCostEveryEpoch ? network.Cost(testData) : -1) }, Accuracy: { network.Validate(testData, (a, o) => ValidateSample(a, o)) } / { testData.Count() }");
            }, onBatch);

            network.Save(@"C:\Users\hadis\source\repos\NeuralNetworksAndDeepLearning\Demo\nets\cross-entropy.mlp");
        }

        public static void VisualizeNetwork(string src, string dest, string networkName)
        {
            var network = NeuralNetwork<CrossEntropy>.Load(Path.Combine(src, networkName + ".mlp"));
            var visualizer = new NetworkVisualizer(network.Weights);
            visualizer.ConstructVisualization(28, dest, networkName);
        }

        private static List<TrainingSample> GetNoise(int inputCount, int outputCount, int count)
        {
            var badOutput = Enumerable.Range(0, outputCount).Select(x => 0.0).ToArray();
            var ret = new List<TrainingSample> {
                new TrainingSample(Enumerable.Range(0, inputCount).Select(x => 0.0).ToArray(), badOutput),
                new TrainingSample(Enumerable.Range(0, inputCount).Select(x => 1.0).ToArray(), badOutput),
            };

            var rand = new Random(DateTime.Now.ToString().GetHashCode());

            for (int i = 2; i < count; i++)
                ret.Add(new TrainingSample(Enumerable.Range(0, inputCount).Select(x => rand.NextDouble()).ToArray(), badOutput));

            return ret;
        }

        private static bool ValidateSample(double[] activations, double[] output)
        {
            double maxActivation = double.MinValue;
            int maxActivationIndex = -2, maxOutputIndex = -1;
            for (int i = 0; i < activations.Length; i++)
            {
                if (activations[i] > maxActivation) (maxActivation, maxActivationIndex) = (activations[i], i);
                if (output[i] == 1.0) maxOutputIndex = i;
            }
            if (maxOutputIndex < 0 || maxActivationIndex < 0) throw new Exception("Yo what the fuck");
            return maxActivationIndex == maxOutputIndex;
        }

        private static TrainingSample[] FetchData(string labelPath, string imagePath, bool print = false)
        {
            BinaryReader labels = new BinaryReader(new FileStream(labelPath, FileMode.Open));

            int magicLabel = ReadBigInt32(labels);
            int numberOfLabels = ReadBigInt32(labels);

            BinaryReader images = new BinaryReader(new FileStream(imagePath, FileMode.Open));

            int magicNumber = ReadBigInt32(images);
            int numberOfImages = ReadBigInt32(images);
            int width = ReadBigInt32(images);
            int height = ReadBigInt32(images);

            TrainingSample[] ret = new TrainingSample[numberOfImages];

            if (print)
            {
                Console.WriteLine($"magicLabel: { magicLabel }");
                Console.WriteLine($"numberOfLabels: { numberOfLabels }");
                Console.WriteLine();
                Console.WriteLine($"magicLabel: { magicNumber }");
                Console.WriteLine($"numberOfImages: { numberOfImages }");
                Console.WriteLine($"width: { width }");
                Console.WriteLine($"height: { height }");
                Console.WriteLine();
            }

            for (int i = 0; i < numberOfImages; i++)
            {
                var label = labels.ReadByte();
                ret[i] = new TrainingSample(
                    images.ReadBytes(width * height).Select(x => x / 255.0).ToArray(),
                    Enumerable.Range(0, 10).Select(c => c == label ? 1.0 : 0.0).ToArray()
                );
            }

            return ret;
        }

        private static int ReadBigInt32(BinaryReader reader)
        {
            var bytes = reader.ReadBytes(4);
            Array.Reverse(bytes);
            return BitConverter.ToInt32(bytes, 0);
        }
    }
}
