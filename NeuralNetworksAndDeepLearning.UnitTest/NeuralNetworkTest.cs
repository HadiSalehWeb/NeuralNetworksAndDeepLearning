using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;

namespace NeuralNetworksAndDeepLearning.UnitTest
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
            var cost = net.Cost(new List<TrainingSample> { new TrainingSample(input, output) });
            Assert.AreEqual(.5 * (activation - output) * (activation - output), cost);
        }

        [TestMethod]
        public void TestValidate()
        {
            Assert.AreEqual(1, net.Validate(new List<TrainingSample> {
                new TrainingSample(1, Sigmoid(1 * 2 + 3)),
                new TrainingSample(0, Sigmoid(0 * 2 + 2))
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
            var backProp = net.Backpropagate(new TrainingSample(x, y));
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
            var dcdw = net.Backpropagate(new TrainingSample(new double[] { x0, x1 }, new double[] { y0 }));

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

        private static double Sigmoid(double z)
        {
            return 1 / (1 + Math.Exp(-z));
        }

        private static double SigmoidPrime(double z)
        {
            return Sigmoid(z) * (1 - Sigmoid(z));
        }
    }
}
