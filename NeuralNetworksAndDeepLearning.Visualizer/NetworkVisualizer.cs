using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;

namespace NeuralNetworksAndDeepLearning.Visualizer
{
    public enum WeightNomralizationMode
    {
        Absolute,
        Signed
    }

    public class NetworkVisualizer
    {
        public List<double[,]> Weights { get; }
        public NetworkVisualizer(List<double[,]> weights)
        {
            Weights = weights.Select(l => ExtractWeights(l)).ToList();
        }

        public void ConstructVisualization(int imageWidth, string path, string networkName)
        {
            path = AddSuffix(Path.Combine(path, networkName));
            Directory.CreateDirectory(Path.Combine(path, networkName));
            string currentPath;

            currentPath = Path.Combine(path, "Absolute");
            Directory.CreateDirectory(currentPath);
            for (int i = 0; i < Weights.Count; i++)
            {
                Directory.CreateDirectory(Path.Combine(currentPath, $"Layer { i }"));
                SaveImages(VisualizeLayer(i, i == 0 ? imageWidth : 1, WeightNomralizationMode.Absolute), Path.Combine(currentPath, $"Layer { i }"));
            }
            Directory.CreateDirectory(Path.Combine(currentPath, $"Network"));
            SaveImages(VisualizeNetwork(imageWidth, WeightNomralizationMode.Absolute), Path.Combine(currentPath, "Network"));

            currentPath = Path.Combine(path, "Signed");
            Directory.CreateDirectory(currentPath);
            for (int i = 0; i < Weights.Count; i++)
            {
                Directory.CreateDirectory(Path.Combine(currentPath, $"Layer { i }"));
                SaveImages(VisualizeLayer(i, i == 0 ? imageWidth : 1, WeightNomralizationMode.Signed), Path.Combine(currentPath, $"Layer { i }"));
            }
            Directory.CreateDirectory(Path.Combine(currentPath, $"Network"));
            SaveImages(VisualizeNetwork(imageWidth, WeightNomralizationMode.Signed), Path.Combine(currentPath, "Network"));
        }

        private string AddSuffix(string path)
        {
            if (!Directory.Exists(path)) return path;
            int i = 0;
            while (Directory.Exists(path + $" ({ i })")) i++;
            return path + $" ({ i })";
        }

        private void SaveImages(Image<Rgba32>[] images, string path)
        {
            var encoder = new PngEncoder();
            for (int i = 0; i < images.Length; i++)
                images[i].Save(Path.Combine(path, $"Neuron { i }.png"), encoder);
        }

        public Image<Rgba32>[] VisualizeLayer(int layer, int imageWidth, WeightNomralizationMode mode)
        {
            return Enumerable
                .Range(0, Weights[layer].GetLength(0))
                .Select(i =>
                    ToImage(Normalize(Square(ExtractNeuronConnections(Weights[layer], i), imageWidth), mode), mode)
                ).ToArray();
        }

        public Image<Rgba32>[] VisualizeNetwork(int imageWidth, WeightNomralizationMode mode)
        {
            return Enumerable.Range(1, Weights.Count - 1)
                .Aggregate(
                    Enumerable.Range(0, Weights[0].GetLength(0))
                    .Select(i =>
                        Square(ExtractNeuronConnections(Weights[0], i), imageWidth)
                    ).ToArray()
                , (squares, layerIndex) =>
                    Enumerable
                    .Range(0, Weights[layerIndex].GetLength(0))
                    .Select(j =>
                        Enumerable.Range(0, Weights[layerIndex].GetLength(1))
                        .Aggregate(new double[squares[j].GetLength(0), squares[j].GetLength(1)], (a, i) =>
                            Add(a, Scale(squares[i], Weights[layerIndex][j, i]))
                        )
                    ).ToArray()
                ).Select(mat =>
                    ToImage(Normalize(mat, mode), mode)
                ).ToArray();
        }

        private static Image<Rgba32> ToImage(double[,] mat, WeightNomralizationMode mode)
        {
            Image<Rgba32> ret = new Image<Rgba32>(mat.GetLength(0), mat.GetLength(1));

            for (int i = 0; i < mat.GetLength(0); i++)
                for (int j = 0; j < mat.GetLength(1); j++)
                {
                    float r = mode == WeightNomralizationMode.Absolute ? (float)mat[i, j] : mat[i, j] < 0 ? (float)-mat[i, j] : 0f;
                    float g = mode == WeightNomralizationMode.Absolute ? (float)mat[i, j] : mat[i, j] > 0 ? (float)mat[i, j] : 0f;
                    float b = mode == WeightNomralizationMode.Absolute ? (float)mat[i, j] : 0;
                    ret[i, j] = new Rgba32(r, g, b);
                }

            return ret;
        }

        private static double[,] Square(double[] weights, int imageWidth)
        {
            double[,] mat = new double[imageWidth, weights.Length / imageWidth];

            for (int i = 0; i < weights.Length; i++)
                mat[i % imageWidth, i / imageWidth] = weights[i];

            return mat;
        }

        private static double[,] Normalize(double[,] mat, WeightNomralizationMode mode)
        {
            return mode == WeightNomralizationMode.Absolute ? NormalizeAbsolute(mat) : NormalizeSigned(mat);
        }

        private static double[,] NormalizeSigned(double[,] mat)
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

        private static double[,] NormalizeAbsolute(double[,] mat)
        {

            double maxVal = double.MinValue, minVal = double.MaxValue;
            for (int i = 0; i < mat.GetLength(0); i++)
                for (int j = 0; j < mat.GetLength(1); j++)
                {
                    if (mat[i, j] > maxVal)
                        maxVal = mat[i, j];
                    if (mat[i, j] < minVal)
                        minVal = mat[i, j];
                }

            var ret = new double[mat.GetLength(0), mat.GetLength(1)];

            double scale = maxVal - minVal;

            for (int i = 0; i < mat.GetLength(0); i++)
                for (int j = 0; j < mat.GetLength(1); j++)
                    ret[i, j] = (mat[i, j] - minVal) / scale;

            return ret;
        }

        private static double[,] Add(double[,] mat1, double[,] mat2)
        {
            var ret = new double[mat1.GetLength(0), mat1.GetLength(1)];

            for (int i = 0; i < mat1.GetLength(0); i++)
                for (int j = 0; j < mat1.GetLength(1); j++)
                    ret[i, j] = mat1[i, j] + mat2[i, j];

            return ret;
        }

        private static double[,] Scale(double[,] mat, double value)
        {
            var ret = new double[mat.GetLength(0), mat.GetLength(1)];

            for (int i = 0; i < mat.GetLength(0); i++)
                for (int j = 0; j < mat.GetLength(1); j++)
                    ret[i, j] = mat[i, j] * value;

            return ret;
        }

        private static double[,] ExtractWeights(double[,] weightsAndBiases)
        {
            var ret = new double[weightsAndBiases.GetLength(0), weightsAndBiases.GetLength(1) - 1];

            for (int i = 0; i < weightsAndBiases.GetLength(0); i++)
                for (int j = 0; j < weightsAndBiases.GetLength(1) - 1; j++)
                    ret[i, j] = weightsAndBiases[i, j];

            return ret;
        }

        private static double[] ExtractNeuronConnections(double[,] weights, int targetNeuron)
        {
            double[] ret = new double[weights.GetLength(1)];

            for (int i = 0; i < weights.GetLength(1); i++)
                ret[i] = weights[targetNeuron, i];

            return ret;
        }
    }
}
