using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System;

namespace NeuralNetworksAndDeepLearning.Visualizer
{
    public static class Util
    {

        public static Image<Rgba32> ToImage(float[,] mat, WeightNomralizationMode mode)
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

        public static float[,] Square(float[] weights, int imageWidth)
        {
            float[,] mat = new float[imageWidth, weights.Length / imageWidth];

            for (int i = 0; i < weights.Length; i++)
                mat[i % imageWidth, i / imageWidth] = weights[i];

            return mat;
        }

        public static float[,] Normalize(float[,] mat, WeightNomralizationMode mode)
        {
            return mode == WeightNomralizationMode.Absolute ? NormalizeAbsolute(mat) : NormalizeSigned(mat);
        }

        public static float[,] NormalizeSigned(float[,] mat)
        {
            var maxVal = float.MinValue;
            for (int i = 0; i < mat.GetLength(0); i++)
                for (int j = 0; j < mat.GetLength(1); j++)
                    if (Math.Abs(mat[i, j]) > maxVal)
                        maxVal = Math.Abs(mat[i, j]);

            var ret = new float[mat.GetLength(0), mat.GetLength(1)];

            for (int i = 0; i < mat.GetLength(0); i++)
                for (int j = 0; j < mat.GetLength(1); j++)
                    ret[i, j] = mat[i, j] / maxVal;

            return ret;
        }

        public static float[,] NormalizeAbsolute(float[,] mat)
        {

            float maxVal = float.MinValue, minVal = float.MaxValue;
            for (int i = 0; i < mat.GetLength(0); i++)
                for (int j = 0; j < mat.GetLength(1); j++)
                {
                    if (mat[i, j] > maxVal)
                        maxVal = mat[i, j];
                    if (mat[i, j] < minVal)
                        minVal = mat[i, j];
                }

            var ret = new float[mat.GetLength(0), mat.GetLength(1)];

            float scale = maxVal - minVal;

            for (int i = 0; i < mat.GetLength(0); i++)
                for (int j = 0; j < mat.GetLength(1); j++)
                    ret[i, j] = (mat[i, j] - minVal) / scale;

            return ret;
        }

        public static float[,] Add(float[,] mat1, float[,] mat2)
        {
            var ret = new float[mat1.GetLength(0), mat1.GetLength(1)];

            for (int i = 0; i < mat1.GetLength(0); i++)
                for (int j = 0; j < mat1.GetLength(1); j++)
                    ret[i, j] = mat1[i, j] + mat2[i, j];

            return ret;
        }

        public static float[] Add(float[] vec1, float[] vec2)
        {
            var ret = new float[vec1.Length];

            for (int i = 0; i < vec1.GetLength(0); i++)
                ret[i] = vec1[i] + vec2[i];

            return ret;
        }

        public static float[,] Scale(float[,] mat, float value)
        {
            var ret = new float[mat.GetLength(0), mat.GetLength(1)];

            for (int i = 0; i < mat.GetLength(0); i++)
                for (int j = 0; j < mat.GetLength(1); j++)
                    ret[i, j] = mat[i, j] * value;

            return ret;
        }

        public static float[,] ExtractWeights(float[,] weightsAndBiases)
        {
            var ret = new float[weightsAndBiases.GetLength(0), weightsAndBiases.GetLength(1) - 1];

            for (int i = 0; i < weightsAndBiases.GetLength(0); i++)
                for (int j = 0; j < weightsAndBiases.GetLength(1) - 1; j++)
                    ret[i, j] = weightsAndBiases[i, j];

            return ret;
        }

        public static float[] ExtractNeuronConnections(float[,] weights, int targetNeuron)
        {
            float[] ret = new float[weights.GetLength(1)];

            for (int i = 0; i < weights.GetLength(1); i++)
                ret[i] = weights[targetNeuron, i];

            return ret;
        }
    }
}
