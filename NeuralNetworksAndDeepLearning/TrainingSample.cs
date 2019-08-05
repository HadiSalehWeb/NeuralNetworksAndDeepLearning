namespace NeuralNetworksAndDeepLearning.Convolutional
{
    public struct TrainingSample
    {
        public float[] Input { get; }
        public float[] Output { get; }

        public TrainingSample(float[] input, float[] output)
        {
            Input = input;
            Output = output;
        }

        public TrainingSample(float input, float output)
        {
            Input = new float[] { input };
            Output = new float[] { output };
        }
    }
}
