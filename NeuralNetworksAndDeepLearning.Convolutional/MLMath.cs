using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworksAndDeepLearning.Convolutional
{
    public static class MLMath
    {
        public static float Sigmoid(float z)
        {
            return (float)(1f / (1f + Math.Exp(-z)));
        }

        public static float SigmoidPrime(float z)
        {
            return Sigmoid(z) * (1 - Sigmoid(z));
        }

        public static float Tanh(float z)
        {
            var e = Math.Exp(2 * z);
            return (float)((e - 1) / (e + 1));
        }

        public static float TanhPrime(float z)
        {
            return 1 - Tanh(z) * Tanh(z);
        }

        public static float ReLU(float z)
        {
            return z <= 0 ? 0 : z;
        }

        public static float ReLUPrime(float z)
        {
            return z <= 0 ? 0 : 1;
        }

        public static List<T> Shuffle<T>(this IEnumerable<T> data, Random rand)
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

        public static List<T> Shuffle<T>(this IEnumerable<T> data)
        {
            return Shuffle(data, new Random(DateTime.Now.ToString().GetHashCode()));
        }

        public static T[] Shuffle<T>(this T[] data, Random rand)
        {
            List<T> ts = new List<T>(data);
            T[] result = new T[data.Length];
            int i = 0;

            while (ts.Any())
            {
                int r = rand.Next(0, ts.Count);
                result[i++] = ts[r];
                ts.RemoveAt(r);
            }

            return result;
        }

        public static T[] Shuffle<T>(this T[] data)
        {
            return Shuffle(data, new Random(DateTime.Now.ToString().GetHashCode()));
        }

        public static float Gaussian(Random rand, float mean, float standardDeviation)
        {
            return (float)(mean + standardDeviation * Math.Sqrt(-2.0 * Math.Log(1.0 - rand.NextDouble())) * Math.Sin(2.0 * Math.PI * (1.0 - rand.NextDouble())));
        }

        public static int ArgMax(IList<float> list)
        {
            int index = -1;
            float val = float.MinValue;

            for (int i = 0; i < list.Count; i++)
                if (list[i] > val)
                    (val, index) = (list[i], i);

            return index;
        }

        internal static float[] Activate(float[] weightedInput, Func<float, float> activationFunction)
        {
            float[] activation = new float[weightedInput.Length];

            for (int i = 0; i < weightedInput.Length; i++)
                activation[i] = activationFunction(weightedInput[i]);

            return activation;
        }
    }
}
