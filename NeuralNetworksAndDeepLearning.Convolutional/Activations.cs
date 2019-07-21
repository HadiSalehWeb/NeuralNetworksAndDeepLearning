using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetworksAndDeepLearning.Convolutional
{
    public static class Activations
    {
        public static (Func<float, float>, Func<float, float>) Sigmoid =
            (SigmoidFun, z => SigmoidFun(z) * (1 - SigmoidFun(z)));
        public static (Func<float, float>, Func<float, float>) Tanh =
            (TanhFun, z => 1 - TanhFun(z) * TanhFun(z));
        public static (Func<float, float>, Func<float, float>) ReLU =
            (z => z <= 0 ? 0 : z, z => z <= 0 ? 0 : 1);

        private static float SigmoidFun(float z)
        {
            return (float)(1f / (1f + Math.Exp(-z)));
        }
        private static float TanhFun(float z)
        {
            var e = Math.Exp(2 * z);
            return (float)((e - 1) / (e + 1));
        }
    }
}
