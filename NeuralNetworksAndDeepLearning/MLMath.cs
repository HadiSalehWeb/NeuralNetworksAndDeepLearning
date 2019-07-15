using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworksAndDeepLearning
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
            return Shuffle(data, new Random());
        }

        public static float Gaussian(Random rand, float mean, float standardDeviation)
        {
            return (float)(mean + standardDeviation * Math.Sqrt(-2.0 * Math.Log(1.0 - rand.NextDouble())) * Math.Sin(2.0 * Math.PI * (1.0 - rand.NextDouble())));
        }
    }
}
