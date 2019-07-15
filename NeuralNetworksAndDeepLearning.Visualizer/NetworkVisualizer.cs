using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;
using static NeuralNetworksAndDeepLearning.Visualizer.Util;

namespace NeuralNetworksAndDeepLearning.Visualizer
{
    public enum WeightNomralizationMode
    {
        Absolute,
        Signed
    }

    public class NetworkVisualizer
    {
        public List<float[,]> Weights { get; }
        public NetworkVisualizer(List<float[,]> weights)
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
                        .Aggregate(new float[squares[j].GetLength(0), squares[j].GetLength(1)], (a, i) =>
                            Add(a, Scale(squares[i], Weights[layerIndex][j, i]))
                        )
                    ).ToArray()
                ).Select(mat =>
                    ToImage(Normalize(mat, mode), mode)
                ).ToArray();
        }
    }
}
