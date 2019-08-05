using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using NeuralNetworksAndDeepLearning.Visualizer;
using SixLabors.ImageSharp;
using NeuralNetworksAndDeepLearning.Convolutional;

namespace NeuralNetworksAndDeepLearning.Demo
{
    class Program
    {
        const int BATCH_SIZE = 10;
        const int EPOCH_COUNT = 10;
        const string TRAINING_IMAGES_PATH = @"C:\Users\hadis\source\repos\NeuralNetworksAndDeepLearning\NeuralNetworksAndDeepLearning.Demo\Training Data\train-images.idx3-ubyte";
        const string TRAINING_LABELS_PATH = @"C:\Users\hadis\source\repos\NeuralNetworksAndDeepLearning\NeuralNetworksAndDeepLearning.Demo\Training Data\train-labels.idx1-ubyte";
        const string TEST_IMAGES_PATH = @"C:\Users\hadis\source\repos\NeuralNetworksAndDeepLearning\NeuralNetworksAndDeepLearning.Demo\Training Data\t10k-images.idx3-ubyte";
        const string TEST_LABELS_PATH = @"C:\Users\hadis\source\repos\NeuralNetworksAndDeepLearning\NeuralNetworksAndDeepLearning.Demo\Training Data\t10k-labels.idx1-ubyte";
        const int INPUT_COUNT = 784;
        const int HIDDEN_COUNT = 30;
        const int OUTPUT_COUNT = 10;
        const float LEARNING_RATE = 0.1f;
        const float REGULARIZATION_RATE = 5f;

        static void Main(string[] args)
        {
            TrainConvOnMnistAndSave();
            //VisualizeNetwork(
            //    @"c:\users\hadis\source\repos\neuralnetworksanddeeplearning\demo\nets",
            //    @"c:\users\hadis\source\repos\neuralnetworksanddeeplearning\demo\visualizations",
            //    @"2-hidden"
            //);
            //EnhanceInput(@"C:\Users\hadis\source\repos\NeuralNetworksAndDeepLearning\Demo", "cross-entropy", new float[] { 1f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f }, 10, 1000);
            //var network = new NeuralNetwork(new List<Layer>
            //{
            //    new InputLayer2d(28, 28),
            //    new ConvolutionalLayer2d(28, 28)
            //});
        }

        public static void EnhanceInput(string path, string networkName, float[] desiredOutput, int epochs, int iterations)
        {
            var net = NeuralNetwork<CrossEntropy>.Load(Path.Combine(path, "nets", networkName + ".mlp"));
            var dist = Path.Combine(path, "Visualizations", networkName, string.Join('_', desiredOutput.Select(x => x.ToString())));
            Directory.CreateDirectory(dist);
            var rand = new Random(DateTime.Now.ToString().GetHashCode());
            var layer = Enumerable.Range(0, net.Weights[0].GetLength(1) - 1).Select(x => (float)rand.NextDouble()).ToArray();
            var encoder = new SixLabors.ImageSharp.Formats.Png.PngEncoder();
            Util.ToImage(Util.Square(layer, 28), WeightNomralizationMode.Absolute).Save(Path.Combine(dist, "Original.png"), encoder);
            var extractor = new FeatureExtractor<CrossEntropy>(net);

            for (int i = 0; i < epochs; i++)
            {
                for (int j = 0; j < iterations; j++)
                    layer = extractor.Enhance(layer, desiredOutput);
                Console.WriteLine($"Finished epoch { i }");
                Util.ToImage(Util.Square(layer, 28), WeightNomralizationMode.Absolute).Save(Path.Combine(dist, $"{ i }.png"), encoder);
            }
            Console.WriteLine(string.Join(", ", net.Feedforward(layer)));
        }

        //public static void TrainOnMnistAndSave(bool printBatchProgress = false, bool printCostEveryEpoch = false, bool printFetchStats = false)
        //{
        //    var network = new NeuralNetwork<CrossEntropy>(new List<int>() { INPUT_COUNT, HIDDEN_COUNT, HIDDEN_COUNT, OUTPUT_COUNT }, Regularization.L2);

        //    var rawData = FetchData(TRAINING_LABELS_PATH, TRAINING_IMAGES_PATH, printFetchStats);
        //    var trainingData = rawData.Take(50000).ToArray();
        //    var validationData = rawData.Skip(50000).ToArray();
        //    var testData = FetchData(TEST_LABELS_PATH, TEST_IMAGES_PATH, printFetchStats);
        //    Action<int> onBatch = null;

        //    if (printBatchProgress)
        //    {
        //        var batchCount = trainingData.Length / BATCH_SIZE;
        //        onBatch = i =>
        //        {
        //            Console.WriteLine($"Batch progress: { i }/{ batchCount }");
        //            Console.SetCursorPosition(0, Console.CursorTop - 1);
        //        };
        //    }
        //    var accuracy = network.Validate(testData, (a, o) => ValidateSample(a, o));
        //    Console.WriteLine($"Starting Cost: { (printCostEveryEpoch ? network.Cost(testData) : -1) }, Accuracy: { accuracy } / { testData.Count() }, { 100.0 * accuracy / testData.Count() }%");

        //    network.SGD(trainingData, EPOCH_COUNT, BATCH_SIZE, LEARNING_RATE, REGULARIZATION_RATE, i =>
        //    {
        //        accuracy = network.Validate(testData, (a, o) => ValidateSample(a, o));
        //        Console.WriteLine($"Finished epoch { i }. Cost: { (printCostEveryEpoch ? network.Cost(testData) : -1) }, Accuracy: { accuracy } / { testData.Count() }, { 100.0 * accuracy / testData.Count() }%");
        //    }, onBatch);

        //    network.Save(@"C:\Users\hadis\source\repos\NeuralNetworksAndDeepLearning\Demo\nets\test.mlp");
        //}

        public static void TrainConvOnMnistAndSave(bool printBatchProgress = false, bool printCostEveryEpoch = false, bool printFetchStats = false)
        {
            //var network = new NeuralNetwork<CrossEntropy>(new List<int>() { INPUT_COUNT, HIDDEN_COUNT, HIDDEN_COUNT, OUTPUT_COUNT }, Regularization.L2);
            var network = new NeuralNetwork(INPUT_COUNT, new Softmax(OUTPUT_COUNT), new Convolutional.Convolutional(1, 1, 5, 1, 1, 28, 28, Activations.Sigmoid));
            //var network = new NeuralNetwork(INPUT_COUNT, new Softmax(OUTPUT_COUNT), new FullyConnectedHidden(30, Activations.Sigmoid));

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
            var accuracy = network.Validate(testData, (a, o) => ValidateSample(a, o));
            Console.WriteLine($"Starting Cost: { (printCostEveryEpoch ? network.Cost(testData) : -1) }, Accuracy: { accuracy } / { testData.Count() }, { 100.0 * accuracy / testData.Count() }%");

            network.SGD(trainingData, EPOCH_COUNT, BATCH_SIZE, LEARNING_RATE,/* REGULARIZATION_RATE, */i =>
            {
                accuracy = network.Validate(testData, (a, o) => ValidateSample(a, o));
                Console.WriteLine($"Finished epoch { i }. Cost: { (printCostEveryEpoch ? network.Cost(testData) : -1) }, Accuracy: { accuracy } / { testData.Count() }, { 100.0 * accuracy / testData.Count() }%");
            }, onBatch);

            //network.Save(@"C:\Users\hadis\source\repos\NeuralNetworksAndDeepLearning\Demo\nets\test.mlp");
        }
        public static void XOR(bool printBatchProgress = false, bool printCostEveryEpoch = false, bool printFetchStats = false)
        {
            var network = new NeuralNetwork(
                2,
                new FullyConnectedOutput(1, FullyConnectedOutput.CostFunction.CrossEntropy),
                new FullyConnectedHidden(2, Activations.Sigmoid)
            );

            var trainingData = new Convolutional.TrainingSample[] {
                new Convolutional.TrainingSample(new float[]{ 0, 0 }, new float[]{ 0 }),
                new Convolutional.TrainingSample(new float[]{ 0, 1 }, new float[]{ 1 }),
                new Convolutional.TrainingSample(new float[]{ 1, 0 }, new float[]{ 1 }),
                new Convolutional.TrainingSample(new float[]{ 1, 1 }, new float[]{ 0 }),
            };
            var testData = trainingData;
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
            var accuracy = network.Validate(testData, (a, o) => ValidateSample(a, o));
            Console.WriteLine($"Starting Cost: { (printCostEveryEpoch ? network.Cost(testData) : -1) }, Accuracy: { accuracy } / { testData.Count() }, { 100.0 * accuracy / testData.Count() }%");

            network.SGD(trainingData, 1000, 4, 10,/* REGULARIZATION_RATE, */i =>
            {
                accuracy = network.Validate(testData, (a, o) => ValidateSample(a, o));
                Console.WriteLine($"Finished epoch { i }. Cost: { (printCostEveryEpoch ? network.Cost(testData) : -1) }, Accuracy: { accuracy } / { testData.Count() }, { 100.0 * accuracy / testData.Count() }%");
            }, onBatch);

            //network.Save(@"C:\Users\hadis\source\repos\NeuralNetworksAndDeepLearning\Demo\nets\test.mlp");
        }

        public static void VisualizeNetwork(string src, string dest, string networkName)
        {
            var network = NeuralNetwork<CrossEntropy>.Load(Path.Combine(src, networkName + ".mlp"));
            var visualizer = new NetworkVisualizer(network.Weights);
            visualizer.ConstructVisualization(28, dest, networkName);
        }

        private static List<TrainingSample> GetNoise(int inputCount, int outputCount, int count)
        {
            var badOutput = Enumerable.Range(0, outputCount).Select(x => 0f).ToArray();
            var ret = new List<TrainingSample> {
                new TrainingSample(Enumerable.Range(0, inputCount).Select(x => 0f).ToArray(), badOutput),
                new TrainingSample(Enumerable.Range(0, inputCount).Select(x => 1f).ToArray(), badOutput),
            };

            var rand = new Random(DateTime.Now.ToString().GetHashCode());

            for (int i = 2; i < count; i++)
                ret.Add(new TrainingSample(Enumerable.Range(0, inputCount).Select(x => (float)rand.NextDouble()).ToArray(), badOutput));

            return ret;
        }

        private static bool ValidateSample(float[] activations, float[] output)
        {
            //return activations[0] > .5 && output[0] > .5 || activations[0] < .5 && output[0] < .5;
            float maxActivation = float.MinValue;
            int maxActivationIndex = -2, maxOutputIndex = -1;
            for (int i = 0; i < activations.Length; i++)
            {
                if (activations[i] > maxActivation) (maxActivation, maxActivationIndex) = (activations[i], i);
                if (output[i] == 1.0) maxOutputIndex = i;
            }
            if (maxOutputIndex < 0 || maxActivationIndex < 0) throw new Exception("Yo what the fuck");
            return maxActivationIndex == maxOutputIndex;
        }

        private static Convolutional.TrainingSample[] FetchData(string labelPath, string imagePath, bool print = false)
        {
            BinaryReader labels = new BinaryReader(new FileStream(labelPath, FileMode.Open));

            int magicLabel = ReadBigInt32(labels);
            int numberOfLabels = ReadBigInt32(labels);

            BinaryReader images = new BinaryReader(new FileStream(imagePath, FileMode.Open));

            int magicNumber = ReadBigInt32(images);
            int numberOfImages = ReadBigInt32(images);
            int width = ReadBigInt32(images);
            int height = ReadBigInt32(images);

            Convolutional.TrainingSample[] ret = new Convolutional.TrainingSample[numberOfImages];

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
                ret[i] = new Convolutional.TrainingSample(
                    images.ReadBytes(width * height).Select(x => x / 255f).ToArray(),
                    Enumerable.Range(0, 10).Select(c => c == label ? 1f : 0f).ToArray()
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
