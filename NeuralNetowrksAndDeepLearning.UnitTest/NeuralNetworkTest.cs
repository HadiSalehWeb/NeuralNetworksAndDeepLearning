using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;

namespace NeuralNetowrksAndDeepLearning.UnitTest
{
    [TestClass]
    public class NeuralNetworkTest
    {
        private NeuralNetwork net;

        [TestInitialize]
        public void Init()
        {
            net = new NeuralNetwork(new List<double[,]>
            {
                new double[1, 2]
                {
                    { 2, 3 }
                }
            });
        }

        [TestMethod]
        public void TestConstructor()
        {
            Assert.AreEqual(2, net.LayerCount);
            Assert.AreEqual(1, net.NeuronCount[0]);
            Assert.AreEqual(1, net.NeuronCount[1]);
            Assert.AreEqual(2, net.Weights[0][0, 0]);
            Assert.AreEqual(3, net.Weights[0][0, 1]);
        }

        [TestMethod]
        public void TestFeedforward()
        {
            var activation = net.Feedforward(new double[] { 1 });
            Assert.AreEqual(1, activation.Length);
            Assert.AreEqual(Sigmoid(1 * 2 + 3), activation[0]);
        }

        [TestMethod]
        public void TestCost()
        {
            double input = 1, output = Sigmoid(1 * 4 + 2), activation = net.Feedforward(new double[] { input })[0];
            var cost = net.Cost(new List<ITrainingSample> { new TestTrainingSample(input, output) });
            Assert.AreEqual(.5 * (activation - output) * (activation - output), cost);
        }

        [TestMethod]
        public void TestValidate()
        {
            Assert.AreEqual(1, net.Validate(new List<ITrainingSample> {
                new TestTrainingSample(1, Sigmoid(1 * 2 + 3)),
                new TestTrainingSample(0, Sigmoid(0 * 2 + 2))
            }, (a, b) => a[0] == b[0]));
        }

        [TestMethod]
        public void TestBackpropagate()
        {
            double x = 1,
                y = 0,
                w = net.Weights[0][0, 0],
                b = net.Weights[0][0, 1],
                z = x * w + b,
                a = Sigmoid(z),
                dcdb = SigmoidPrime(z) * (a - y);
            var backProp = net.Backpropagate(new TestTrainingSample(x, y));
            Assert.AreEqual(x * dcdb, backProp[0][0, 0]);
            Assert.AreEqual(dcdb, backProp[0][0, 1]);
        }

        [TestMethod]
        public void TestAnotherBackprop()
        {
            double x0 = .2, x1 = .4,
                y0 = .6,
                w000 = 1.2, w001 = 2.0, b00 = 1,
                w010 = -0.324, w011 = 0.812, b01 = 2,
                w020 = -1.563, w021 = 5.2, b02 = 3,
                w100 = 0.1234, w101 = 3.234, w102 = -3.836, b10 = 7.12,
                a00 = x0, a01 = x1,
                z10 = a00 * w000 + a01 * w001 + b00, a10 = Sigmoid(z10),
                z11 = a00 * w010 + a01 * w011 + b01, a11 = Sigmoid(z11),
                z12 = a00 * w020 + a01 * w021 + b02, a12 = Sigmoid(z12),
                z20 = a10 * w100 + a11 * w101 + a12 * w102 + b10, a20 = Sigmoid(z20),
                dcda20 = a20 - y0,
                da20az20 = SigmoidPrime(z20),
                dcdw100 = dcda20 * da20az20 * a10,
                dcdw101 = dcda20 * da20az20 * a11,
                dcdw102 = dcda20 * da20az20 * a12,
                dcdb10 = dcda20 * da20az20 * 1,
                dcda10 = dcda20 * da20az20 * w100,
                dcda11 = dcda20 * da20az20 * w101,
                dcda12 = dcda20 * da20az20 * w102,
                da10dz10 = SigmoidPrime(z10), da11dz11 = SigmoidPrime(z11), da12dz12 = SigmoidPrime(z12),
                dcdw000 = dcda10 * da10dz10 * a00,
                dcdw001 = dcda10 * da10dz10 * a01,
                dcdb00 = dcda10 * da10dz10 * 1,
                dcdw010 = dcda11 * da11dz11 * a00,
                dcdw011 = dcda11 * da11dz11 * a01,
                dcdb01 = dcda11 * da11dz11 * 1,
                dcdw020 = dcda12 * da12dz12 * a00,
                dcdw021 = dcda12 * da12dz12 * a01,
                dcdb02 = dcda12 * da12dz12 * 1;

            var net = new NeuralNetwork(new List<double[,]>
            {
                new double[,] {
                    { w000, w001, b00 },
                    { w010, w011, b01 },
                    { w020, w021, b02 },
                },
                new double[,] {
                    { w100, w101, w102, b10 }
                },
            });

            var (weightedInputs, activations) = net.GetWeightedInputsAndActivations(new double[] { x0, x1 });
            var dcdw = net.Backpropagate(new TestTrainingSample(new double[] { x0, x1 }, new double[] { y0 }));

            Assert.AreEqual(3, weightedInputs.Count);
            Assert.AreEqual(3, weightedInputs[1].Length);
            Assert.AreEqual(z10, weightedInputs[1][0]);
            Assert.AreEqual(z11, weightedInputs[1][1]);
            Assert.AreEqual(z12, weightedInputs[1][2]);
            Assert.AreEqual(1, weightedInputs[2].Length);
            Assert.AreEqual(z20, weightedInputs[2][0]);

            Assert.AreEqual(3, activations.Count);
            Assert.AreEqual(2, activations[0].Length);
            Assert.AreEqual(a00, activations[0][0]);
            Assert.AreEqual(a01, activations[0][1]);
            Assert.AreEqual(3, activations[1].Length);
            Assert.AreEqual(a10, activations[1][0]);
            Assert.AreEqual(a11, activations[1][1]);
            Assert.AreEqual(a12, activations[1][2]);
            Assert.AreEqual(1, activations[2].Length);
            Assert.AreEqual(a20, activations[2][0]);

            Assert.AreEqual(2, dcdw.Count);
            Assert.AreEqual(3, dcdw[0].GetLength(0));
            Assert.AreEqual(3, dcdw[0].GetLength(1));
            Assert.AreEqual(dcdw000, dcdw[0][0, 0]);
            Assert.AreEqual(dcdw001, dcdw[0][0, 1]);
            Assert.AreEqual(dcdb00, dcdw[0][0, 2]);
            Assert.AreEqual(dcdw010, dcdw[0][1, 0]);
            Assert.AreEqual(dcdw011, dcdw[0][1, 1]);
            Assert.AreEqual(dcdb01, dcdw[0][1, 2]);
            Assert.AreEqual(dcdw020, dcdw[0][2, 0]);
            Assert.AreEqual(dcdw021, dcdw[0][2, 1]);
            Assert.AreEqual(dcdb02, dcdw[0][2, 2]);
            Assert.AreEqual(1, dcdw[1].GetLength(0));
            Assert.AreEqual(4, dcdw[1].GetLength(1));
            Assert.AreEqual(dcdw100, dcdw[1][0, 0]);
            Assert.AreEqual(dcdw101, dcdw[1][0, 1]);
            Assert.AreEqual(dcdw102, dcdw[1][0, 2]);
            Assert.AreEqual(dcdb10, dcdw[1][0, 3]);
        }

        public void Test()
        {
            var net2 = new NeuralNetwork(new List<double[,]> {
                new double[,] {
                    { 0.57165926, 0.08601931,-1.30309699, 1.32204796e+00 },
                    {-0.65271343,-1.89909947,-0.13885406, 1.54735882e-03 },
                    { 0.20234623, 0.31274258, 1.3006446 , 4.83497936e-01 },
                    {-0.63613441,-2.58750023,-0.55782109,-2.28332196e-01 },
                    { 1.46336543, 1.00468728,-0.46975042,-4.82733125e-01 },
                    {-0.3153366 ,-0.12792181,-0.66548699,-1.59718100e-01 },
                    { 0.74572267, 0.94824087, 1.29927483,-5.68493082e-02 },
                    {-0.58894038, 0.41336364,-0.30780699, 2.50667114e-01 },
                    {-1.20760309, 0.43435092,-0.4564507 , 6.23843731e-01 },
                    {-0.31738888, 0.02802328, 1.47057702, 8.99994542e-01 },
                    { 0.21696115, 0.3730925 , 0.35950953, 9.87308623e-01 },
                    {-0.17822286,-1.09532559,-1.01415451, 1.30602229e+00 },
                    { 2.14255559,-1.00320763,-0.32929066, 1.87165650e-01 },
                    {-0.25657647,-0.73631381, 1.21137512, 1.90285493e+00 }
                }, new double[,] {
                    { 0.90982953,-0.04763224, 0.45499072,-0.68421116, 0.06842861,
                     -1.05171463, 0.5565004 ,-0.76273437,-0.94521195,-0.61693557,
                     -0.15964708, 1.1558435 ,-0.52798021,-1.08739114, 0.02565225 },
                    {-0.7290326 , 1.04375309, 1.00207475,-1.11804175,-0.61209612,
                      0.19171488,-1.98759916,-0.26376197, 0.65931088,-1.23391677,
                     -1.24799651,-1.00913517, 0.0876127 ,-0.50833911, 0.40753292 }
                }
            });

            Assert.AreEqual(3, net2.LayerCount);
            Assert.AreEqual(3, net2.NeuronCount[0]);
            Assert.AreEqual(14, net2.NeuronCount[1]);
            Assert.AreEqual(2, net2.NeuronCount[2]);

            net2.UpdateMiniBatch(new List<ITrainingSample> {
                new TestTrainingSample(new double[] { 1, 2, 3 }, new double[]{ 4, 5 })
            }, 3.0);

            Assert.AreEqual(new List<double[,]>
            {
                new double[,] {
                    { 0.74569248, 0.43408573,-0.78099735, 1.49608118 },
                    {-0.6512965 ,-1.89626561,-0.13460327, 0.00296429 },
                    { 0.20825125, 0.32455263, 1.31835966, 0.48940296 },
                    {-0.63682527,-2.58888195,-0.55989367,-0.22902306 },
                    { 1.45836361, 0.99468363,-0.48475588,-0.48773495 },
                    {-0.42343062,-0.34410984,-0.98976904,-0.26781212 },
                    { 0.74648123, 0.94975799, 1.30155051,-0.05609075 },
                    {-0.93875105,-0.28625768,-1.35723898,-0.09914355 },
                    {-1.50187925,-0.15420141,-1.33927919, 0.32956757 },
                    {-0.32658673, 0.00962758, 1.44298347, 0.89079669 },
                    { 0.18967448, 0.31851917, 0.27764953, 0.96002196 },
                    {-0.14866815,-1.03621617,-0.92549038, 1.335577   },
                    { 1.93076503,-1.42678875,-0.96466236,-0.02462491 },
                    {-0.3017285 ,-0.82661788, 1.07591901, 1.85770289 }
                }, new double[,]{
                    { 1.16012306,-0.03362615, 2.27694552,-0.68339266, 1.58762222,
                     -0.93927769, 2.38557715,-0.04291643,-0.48234805, 1.20327587,
                      1.58771542, 1.18555187, 0.09433403, 0.7047022 , 1.85752556},
                    {-0.69316378, 1.04576026, 1.26317362,-1.11792446,-0.39438505,
                      0.20782788,-1.72547966,-0.16060702, 0.72564251,-0.97306773,
                     -0.99758724,-1.00487776, 0.17679468,-0.2515196 , 0.67005318}
                }
            }, net2.Weights);
        }

        private static double Sigmoid(double z)
        {
            return 1 / (1 + Math.Exp(-z));
        }

        private static double SigmoidPrime(double z)
        {
            return Sigmoid(z) * (1 - Sigmoid(z));
        }

        struct TestTrainingSample : ITrainingSample
        {
            public double[] Input { get; }
            public double[] Output { get; }

            public TestTrainingSample(double input, double output)
            {
                Input = new double[] { input };
                Output = new double[] { output };
            }

            public TestTrainingSample(double[] input, double[] output)
            {
                Input = input;
                Output = output;
            }
        }
    }
}
