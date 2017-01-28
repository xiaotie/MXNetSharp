using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace Lenet
{
    using MXNetSharp;

    public class Program
    {
        public static void Main(string[] args)
        {
        }
    }

    public class Lenet
    {
        Context ctx_cpu = new Context(DeviceType.kCPU, 0);
        Context ctx_dev = new Context(DeviceType.kGPU, 0);
        Dictionary<string, NDArray> args_map = new Dictionary<string, NDArray>();
        NDArray train_data;
        NDArray train_label;
        NDArray val_data;
        NDArray val_label;

        public void Run()
        {
            /*
            * LeCun, Yann, Leon Bottou, Yoshua Bengio, and Patrick Haffner.
            * "Gradient-based learning applied to document recognition."
            * Proceedings of the IEEE (1998)
            * */
            /*define the symbolic net*/

            Symbol data = Symbol.Variable("data");
            Symbol data_label = Symbol.Variable("data_label");
            Symbol conv1_w = new Symbol("conv1_w"), conv1_b = new Symbol("conv1_b");
            Symbol conv2_w = new Symbol("conv2_w"), conv2_b = new Symbol("conv2_b");
            Symbol conv3_w = new Symbol("conv3_w"), conv3_b = new Symbol("conv3_b");
            Symbol fc1_w = new Symbol("fc1_w"), fc1_b = new Symbol("fc1_b");
            Symbol fc2_w = new Symbol("fc2_w"), fc2_b = new Symbol("fc2_b");

            Symbol conv1 = Symbol.Convolution("conv1", data, conv1_w, conv1_b, new Shape(5, 5), 20);
            Symbol tanh1 = Symbol.Activation("tanh1", conv1, ActivationActType.tanh);
            Symbol pool1 = Symbol.Pooling("pool1", tanh1, new Shape(2, 2), PoolingPoolType.max,
                false, PoolingPoolingConvention.valid, new Shape(2, 2), new Shape(0,0));

            Symbol conv2 = Symbol.Convolution("conv2", pool1, conv2_w, conv2_b, new Shape(5, 5), 50);
            Symbol tanh2 = Symbol.Activation("tanh2", conv2, ActivationActType.tanh);
            Symbol pool2 = Symbol.Pooling("pool2", tanh2, new Shape(2, 2), PoolingPoolType.max,
              false, PoolingPoolingConvention.valid, new Shape(2, 2), new Shape(0,0));

            Symbol conv3 = Symbol.Convolution("conv3", pool2, conv3_w, conv3_b, new Shape(2, 2), 500);
            Symbol tanh3 = Symbol.Activation("tanh3", conv3, ActivationActType.tanh);
            Symbol pool3 = Symbol.Pooling("pool3", tanh3, new Shape(2, 2), PoolingPoolType.max,
              false, PoolingPoolingConvention.valid, new Shape(1, 1), new Shape(0,0));

            Symbol flatten = Symbol.Flatten("flatten", pool3);
            Symbol fc1 = Symbol.FullyConnected("fc1", flatten, fc1_w, fc1_b, 500);
            Symbol tanh4 = Symbol.Activation("tanh4", fc1, ActivationActType.tanh);
            Symbol fc2 = Symbol.FullyConnected("fc2", tanh4, fc2_w, fc2_b, 10);

            Symbol lenet = Symbol.SoftmaxOutput("softmax", fc2, data_label);

            foreach (String s in lenet.ListArguments())
                Console.WriteLine(s);

            /*setup basic configs*/
            int val_fold = 1;
            int W = 28;
            int H = 28;
            int batch_size = 42;
            int max_epoch = 100000;
            float learning_rate = 0.0001f;
            float weight_decay = 0.0001f;

        }
    }
}
