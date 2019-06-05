using System;
using NeuralNetworksAndDeepLearning;
using System.IO;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System.Collections.Generic;
using System.Linq;

namespace MNISTNetVisualizer
{
    class Program
    {
        static void Main(string[] args)
        {
            var networkName = "30-hidden-noise";
            var netPath = $@"C:\Users\hadis\source\repos\NeuralNetworksAndDeepLearning\Demo\nets";
            var imagesPath = $@"C:\Users\hadis\source\repos\NeuralNetworksAndDeepLearning\MNISTNetVisualizer\images";
            var extension = "mlp";
            var net = NeuralNetwork.Load(Path.Combine(netPath, $"{ networkName }.{ extension }"));
            if (!Directory.Exists(Path.Combine(imagesPath, networkName)))
                Directory.CreateDirectory(Path.Combine(imagesPath, networkName));

            var imageLists = net.Weights.Select(layer => LayerToimages(layer, layer.GetLength(1) - 1 == 28 * 28 ? 28 : 1)).ToList();

            for (int i = 0; i < imageLists.Count; i++)
            {
                if (!Directory.Exists(Path.Combine(imagesPath, networkName, $"Layer_{ i }")))
                    Directory.CreateDirectory(Path.Combine(imagesPath, networkName, $"Layer_{ i }"));

                for (int j = 0; j < imageLists[i].Count; j++)
                    imageLists[i][j].Save(Path.Combine(imagesPath, networkName, $"Layer_{ i }", $"Neuron_{ j }.png"), new SixLabors.ImageSharp.Formats.Png.PngEncoder());
                Console.WriteLine($"Finished layer #{ i }.");
            }

            if (net.Weights.Count > 1)
            {
                if (!Directory.Exists(Path.Combine(imagesPath, networkName, $"Aggregate")))
                    Directory.CreateDirectory(Path.Combine(imagesPath, networkName, $"Aggregate"));

                var aggregate = AggregateLayers(net.Weights, 28).Select(x => ToImage(x)).ToList();

                for (int j = 0; j < aggregate.Count; j++)
                    aggregate[j].Save(Path.Combine(imagesPath, networkName, $"Aggregate", $"Neuron_{ j }.png"), new SixLabors.ImageSharp.Formats.Png.PngEncoder());
                Console.WriteLine("Finished aggregate.");
            }
        }

        private static List<Image<Rgba32>> LayerToimages(double[,] layer, int imageWidth)
        {
            List<Image<Rgba32>> images = new List<Image<Rgba32>>();

            var mat = Normalize(ExtractWeights(layer));
            for (int i = 0; i < mat.GetLength(0); i++)
            {
                Image<Rgba32> image = new Image<Rgba32>(imageWidth, (layer.GetLength(1) - 1) / imageWidth);
                for (int j = 0; j < mat.GetLength(1); j++)
                {
                    var val = (float)mat[i, j];
                    image[j % imageWidth, j / imageWidth] = new Rgba32(val < 0 ? -val : 0, val > 0 ? val : 0, 0);
                }
                images.Add(image);
            }

            return images;
        }

        private static Image<Rgba32> ToImage(double[,] layer)
        {
            Image<Rgba32> image = new Image<Rgba32>(layer.GetLength(0), layer.GetLength(1));
            for (int i = 0; i < layer.GetLength(0); i++)
                for (int j = 0; j < layer.GetLength(1); j++)
                {
                    var val = (float)layer[i, j];
                    image[i, j] = new Rgba32(val < 0 ? -val : 0, val > 0 ? val : 0, 0);
                }
            return image;
        }

        private static List<double[,]> AggregateLayers(List<double[,]> layers, int inputWidth)
        {
            var weights0 = ExtractWeights(layers[0]);
            var seed = new List<double[,]>();

            for (int i = 0; i < weights0.GetLength(0); i++)
            {
                seed.Add(new double[inputWidth, weights0.GetLength(1) / inputWidth]);
                for (int j = 0; j < weights0.GetLength(1); j++)
                    seed[i][j % inputWidth, j / inputWidth] = weights0[i, j];
            }

            for (int l = 1; l < layers.Count; l++)
            {
                var weights = ExtractWeights(layers[l]);
                var temp = new List<double[,]>();
                for (int i = 0; i < weights.GetLength(0); i++)
                {
                    temp.Add(new double[inputWidth, weights0.GetLength(1) / inputWidth]);
                    for (int j = 0; j < weights.GetLength(1); j++)
                        temp[i] = Add(temp[i], Scale(seed[j], weights[i, j]));
                }
                seed = temp;
            }

            return seed.Select(x => Normalize(x)).ToList();
        }

        private static double Sigmoid(double z)
        {
            return 1 / (1 + Math.Exp(-z));
        }

        public static double[,] Normalize(double[,] mat)
        {
            var maxVal = double.MinValue;
            for (int i = 0; i < mat.GetLength(0); i++)
                for (int j = 0; j < mat.GetLength(1); j++)
                    if (Math.Abs(mat[i, j]) > maxVal)
                        maxVal = Math.Abs(mat[i, j]);

            var ret = new double[mat.GetLength(0), mat.GetLength(1)];

            for (int i = 0; i < mat.GetLength(0); i++)
                for (int j = 0; j < mat.GetLength(1); j++)
                    ret[i, j] = mat[i, j] / maxVal;

            return ret;
        }

        public static double[,] Scale(double[,] mat, double value)
        {
            var ret = new double[mat.GetLength(0), mat.GetLength(1)];

            for (int i = 0; i < mat.GetLength(0); i++)
                for (int j = 0; j < mat.GetLength(1); j++)
                    ret[i, j] = mat[i, j] * value;

            return ret;
        }

        public static double[,] Add(double[,] mat1, double[,] mat2)
        {
            var ret = new double[mat1.GetLength(0), mat1.GetLength(1)];

            for (int i = 0; i < mat1.GetLength(0); i++)
                for (int j = 0; j < mat1.GetLength(1); j++)
                    ret[i, j] = mat1[i, j] + mat2[i, j];

            return ret;
        }

        public static double[,] ExtractWeights(double[,] weightsAndBiases)
        {
            var ret = new double[weightsAndBiases.GetLength(0), weightsAndBiases.GetLength(1) - 1];

            for (int i = 0; i < weightsAndBiases.GetLength(0); i++)
                for (int j = 0; j < weightsAndBiases.GetLength(1) - 1; j++)
                    ret[i, j] = weightsAndBiases[i, j];

            return ret;
        }

        public static double[,] ExtractBiases(double[,] weightsAndBiases)
        {
            var ret = new double[weightsAndBiases.GetLength(0), 1];

            for (int i = 0; i < weightsAndBiases.GetLength(0); i++)
                ret[i, 0] = weightsAndBiases[i, weightsAndBiases.GetLength(1) - 1];

            return ret;
        }

        public static double[] Flatten(double[,] mat)
        {
            var ret = new double[mat.GetLength(0) * mat.GetLength(1)];

            for (int i = 0, c = 0; i < mat.GetLength(0); i++)
                for (int j = 0; j < mat.GetLength(1); j++)
                    ret[c++] = mat[i, j];

            return ret;
        }
    }
}
