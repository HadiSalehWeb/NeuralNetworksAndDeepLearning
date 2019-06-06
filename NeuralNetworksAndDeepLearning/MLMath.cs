using System;
using System.Collections.Generic;
using System.Linq;

namespace NeuralNetworksAndDeepLearning
{
    public static class MLMath
    {
        public static double Sigmoid(double z)
        {
            return 1 / (1 + Math.Exp(-z));
        }

        public static double SigmoidPrime(double z)
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
    }
}
