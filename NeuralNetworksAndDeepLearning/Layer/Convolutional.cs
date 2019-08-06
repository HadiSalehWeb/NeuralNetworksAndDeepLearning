using NeuralNetworksAndDeepLearning.Interface;
using NeuralNetworksAndDeepLearning.Model;
using System;
using System.Linq;

namespace NeuralNetworksAndDeepLearning.Layer
{
    public class Convolutional : IHiddenLayer
    {
        public Kernel[] Kernels { get; }
        public int KernelCount { get; }
        public int KernelDepth { get; }
        public int KernelWidth { get; }
        public int KernelHeight { get; }
        //public int StrideX { get; }
        //public int StrideY { get; }
        //public int MaxPoolWidth { get; }
        //public int MaxPoolHeight { get; }
        public int ParameterCount => KernelCount * (KernelWidth * KernelHeight + 1);

        public int InputDepth { get; private set; }
        public int InputWidth { get; }
        public int InputHeight { get; }
        public int InputDimension => InputDepth * InputWidth * InputHeight;

        public int OutputDepth => KernelCount;
        //public int ConvolvedWidth => ((InputWidth - KernelWidth) / StrideX) + 1;
        //public int ConvolvedHeight => ((InputHeight - KernelHeight) / StrideY) + 1;
        //public int PooledWidth => ConvolvedWidth / MaxPoolWidth;
        //public int PooledHeight => ConvolvedHeight / MaxPoolHeight;
        public int OutputWidth => (InputWidth - KernelWidth) + 1;
        public int OutputHeight => (InputHeight - KernelHeight) + 1;
        //public int OutputDimension => OutputDepth * PooledWidth * PooledHeight;
        public int OutputDimension => OutputDepth * OutputWidth * OutputHeight;

        public Func<float, float> ActivationFunction { get; }
        public Func<float, float> ActivationDerivative { get; }

        public Convolutional(int kernelCount, int kernelDepth, int kernelWidth, int kernelHeight, int strideX, int strideY,
            int maxPoolWidth, int maxPoolHeight, int inputWidth, int inputHeight,
            (Func<float, float>, Func<float, float>) activation, Random rand)
        {
            KernelCount = kernelCount;
            KernelDepth = kernelDepth;
            KernelWidth = kernelWidth;
            KernelHeight = kernelHeight;
            //StrideX = strideX;
            //StrideY = strideY;
            //MaxPoolWidth = maxPoolWidth;
            //MaxPoolHeight = maxPoolHeight;
            InputWidth = inputWidth;
            InputHeight = inputHeight;
            (ActivationFunction, ActivationDerivative) = activation;
            Kernels = Enumerable.Range(0, kernelCount).Select(i => new Kernel(kernelDepth, kernelWidth, kernelHeight, rand)).ToArray();
        }
        public Convolutional(int kernelCount, int kernelDepth, int kernelSize, int stride, int maxPoolSize, int inputWidth,
            int inputHeight, (Func<float, float>, Func<float, float>) activation, Random rand)
            : this(kernelCount, kernelDepth, kernelSize, kernelSize, stride, stride, maxPoolSize, maxPoolSize, inputWidth,
                  inputHeight, activation, rand)
        { }
        public Convolutional(int kernelCount, int kernelDepth, int kernelWidth, int kernelHeight, int strideX, int strideY,
            int maxPoolWidth, int maxPoolHeight, int inputWidth, int inputHeight,
            (Func<float, float>, Func<float, float>) activation)
            : this(kernelCount, kernelDepth, kernelWidth, kernelHeight, strideX, strideY, maxPoolWidth, maxPoolHeight,
                  inputWidth, inputHeight, activation, new Random(DateTime.Now.ToString().GetHashCode()))
        { }
        public Convolutional(int kernelCount, int kernelDepth, int kernelSize, int stride, int maxPoolSize, int inputWidth,
            int inputHeight, (Func<float, float>, Func<float, float>) activation)
            : this(kernelCount, kernelDepth, kernelSize, stride, maxPoolSize, inputWidth, inputHeight, activation,
                  new Random(DateTime.Now.ToString().GetHashCode()))
        { }

        public void Initialize(int outputDimensionOfPreviousLayer)
        {
            InputDepth = outputDimensionOfPreviousLayer / (InputWidth * InputHeight);
            if (InputDepth != KernelDepth) throw new Exception("Remove this later");//todo: remove this later
        }

        public float[] GetWeightedInput(float[] input)
        {
            float[] weightedInput = new float[OutputDimension/* * MaxPoolWidth * MaxPoolHeight*/];

            for (int l = 0; l < OutputDepth; l++)
                for (int i = 0; i < OutputWidth; i++)
                    for (int j = 0; j < OutputHeight; j++)
                        weightedInput[l * OutputWidth * OutputHeight + i * OutputHeight + j] = Kernels[l].Filter(input, i/* * StrideX*/, j/* * StrideY*/, InputWidth, InputHeight);

            return weightedInput;

            //return MaxPoolWidth == 1 && MaxPoolHeight == 1 ? weightedInput : MaxPool(weightedInput);
        }

        //public float[] MaxPool(float[] weightedInput)
        //{
        //    float[] pooled = new float[PooledWidth * PooledHeight];

        //    for (int l = 0; l < OutputDepth; l++)
        //        for (int i = 0; i < PooledWidth; i++)
        //            for (int j = 0; j < PooledHeight; j++)
        //            {
        //                float[] pool = new float[MaxPoolWidth * MaxPoolHeight];

        //                for (int x = 0; x < MaxPoolWidth; x++)
        //                    for (int y = 0; y < MaxPoolHeight; y++)
        //                        pool[x * MaxPoolHeight + y] = weightedInput[l * ConvolvedWidth * ConvolvedHeight + (i * MaxPoolWidth + x) * ConvolvedHeight + j * MaxPoolHeight + y];

        //                pooled[i * PooledHeight + j] = pool.Max();
        //            }

        //    return pooled;
        //}

        public float[] GetActivation(float[] weightedInput)
        {
            float[] activation = new float[weightedInput.Length];

            for (int i = 0; i < weightedInput.Length; i++)
                activation[i] = ActivationFunction(weightedInput[i]);

            return activation;
        }

        public float[] Feedforward(float[] input)
        {
            return GetActivation(GetWeightedInput(input));
        }

        public void UpdateParameters(float[] costGradient)
        {
            for (int l = 0, index = 0; l < KernelCount; l++)
            {
                for (int i = 0; i < KernelDepth; i++)
                    for (int j = 0; j < KernelWidth; j++)
                        for (int k = 0; k < KernelHeight; k++, index++)
                            Kernels[l].Weights[i, j, k] += costGradient[index];

                Kernels[l].Bias += costGradient[index++];
            }
        }

        public float[] Backprop(float[] delCostOverDelActivations, IForwardPropData ownForwardPropData, IForwardPropData nextForwardPropData)
        {
            float[] gradient = new float[KernelCount * (KernelDepth * KernelWidth * KernelHeight + 1)];

            for (int l = 0, index = 0; l < KernelCount; l++)
            {
                for (int i = 0; i < KernelDepth; i++)
                    for (int j = 0; j < KernelWidth; j++)
                        for (int k = 0; k < KernelHeight; k++, index++)
                            for (int n = 0; n < OutputWidth; n++)
                                for (int m = 0; m < OutputHeight; m++)
                                    gradient[index] +=
                                        delCostOverDelActivations[l * OutputWidth * OutputHeight + n * OutputHeight + m] *
                                        ActivationDerivative(outWeightedInputs[l * OutputWidth * OutputHeight + n * OutputHeight + m]) *
                                        inActivations[i * KernelWidth * KernelHeight + (n + j) * KernelHeight + m + k];

                for (int n = 0; n < OutputWidth; n++)
                    for (int m = 0; m < OutputHeight; m++)
                        gradient[index] +=
                            delCostOverDelActivations[l * OutputWidth * OutputHeight + n * OutputHeight + m] *
                            ActivationDerivative(outWeightedInputs[l * OutputWidth * OutputHeight + n * OutputHeight + m]);

                index++;
            }

            return gradient;
        }

        public float[] BackpropagateDelCostOverDelActivations(float[] delCostOverDelActivations, float[] outWeightedInputs)
        {
            float[] gradient = new float[InputDimension];

            for (int l = 0; l < KernelCount; l++)
                for (int n = 0; n < OutputWidth; n++)
                    for (int m = 0; m < OutputHeight; m++)
                        for (int i = 0; i < KernelDepth; i++)
                            for (int j = 0; j < KernelWidth; j++)
                                for (int k = 0; k < KernelHeight; k++)
                                    gradient[i * InputWidth * InputHeight + (n + j) * InputHeight + m + k] +=
                                        delCostOverDelActivations[l * OutputWidth * OutputHeight + n * OutputHeight + m] *
                                        ActivationDerivative(outWeightedInputs[l * OutputWidth * OutputHeight + n * OutputHeight + m]) *
                                        Kernels[l].Weights[i, j, k];

            return gradient;
        }

        public float[] BackpropagateDelCostOverDelActivations(float[] delCostOverDelActivations, IForwardPropData ownForwardPropData)
        {
            throw new NotImplementedException();
        }
    }
}
