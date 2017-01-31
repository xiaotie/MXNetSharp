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
            Lenet lenet = new Lenet();
            lenet.Run();
        }
    }

    public unsafe class Lenet
    {
        Context ctx_cpu = new Context(DeviceType.kCPU, 0);
        Context ctx_dev = new Context(DeviceType.kGPU, 0);
        Dictionary<string, NDArray> args_map = new Dictionary<string, NDArray>();
        NDArray train_data;
        NDArray train_label;
        NDArray val_data;
        NDArray val_label;

        private Symbol CreateLenetCpp()
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
                false, PoolingPoolingConvention.valid, new Shape(2, 2), new Shape(0, 0));

            Symbol conv2 = Symbol.Convolution("conv2", pool1, conv2_w, conv2_b, new Shape(5, 5), 50);
            Symbol tanh2 = Symbol.Activation("tanh2", conv2, ActivationActType.tanh);
            Symbol pool2 = Symbol.Pooling("pool2", tanh2, new Shape(2, 2), PoolingPoolType.max,
              false, PoolingPoolingConvention.valid, new Shape(2, 2), new Shape(0, 0));

            Symbol conv3 = Symbol.Convolution("conv3", pool2, conv3_w, conv3_b, new Shape(2, 2), 500);
            Symbol tanh3 = Symbol.Activation("tanh3", conv3, ActivationActType.tanh);
            Symbol pool3 = Symbol.Pooling("pool3", tanh3, new Shape(2, 2), PoolingPoolType.max,
              false, PoolingPoolingConvention.valid, new Shape(1, 1), new Shape(0, 0));

            Symbol flatten = Symbol.Flatten("flatten", pool3);
            Symbol fc1 = Symbol.FullyConnected("fc1", flatten, fc1_w, fc1_b, 500);
            Symbol tanh4 = Symbol.Activation("tanh4", fc1, ActivationActType.tanh);
            Symbol fc2 = Symbol.FullyConnected("fc2", tanh4, fc2_w, fc2_b, 10);

            Symbol lenet = Symbol.SoftmaxOutput("softmax", fc2, data_label);

            System.IO.File.WriteAllText("lenet.json", lenet.ToJSON());
            return lenet;
        }

        private Symbol CreateLenetPython()
        {
            Symbol data = Symbol.Variable("data");
            Symbol data_label = Symbol.Variable("data_label");

            // first conv

            //  mx.symbol.Convolution(data = data, kernel = (5, 5), num_filter = 20)
            Symbol conv1 = new Operator("Convolution").SetParam("kernel", "(5, 5)").SetParam("num_filter", "20").SetData(data).CreateSymbol("conv1");
            // tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
            Symbol tanh1 = new Operator("Activation").SetParam("act_type", "tanh").SetData(conv1).CreateSymbol("tanh1");
            // pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max", kernel = (2,2), stride = (2,2))
            Symbol pool1 = new Operator("Pooling").SetParam("pool_type", "max").SetParam("kernel", "(2, 2)").SetParam("stride", "(2, 2)").SetData(tanh1).CreateSymbol("pool1");

            // second conv
            // conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=50)
            Symbol conv2 = new Operator("Convolution").SetParam("kernel", "(5, 5)").SetParam("num_filter", "50").SetData(pool1).CreateSymbol("conv2");
            // tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
            Symbol tanh2 = new Operator("Activation").SetParam("act_type", "tanh").SetData(conv2).CreateSymbol("tanh2");
            // pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max", kernel = (2,2), stride = (2,2))
            Symbol pool2 = new Operator("Pooling").SetParam("pool_type", "max").SetParam("kernel", "(2, 2)").SetParam("stride", "(2, 2)").SetData(tanh2).CreateSymbol("pool2");

            // first fullc
            // flatten = mx.symbol.Flatten(data=pool2)
            Symbol flatten = new Operator("Flatten").SetData(pool2).CreateSymbol("flatten");
            // fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
            Symbol fc1 = new Operator("FullyConnected").SetParam("num_hidden", "500").SetData(flatten).CreateSymbol("fc1");
            // tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
            Symbol tanh3 = new Operator("Activation").SetParam("act_type", "tanh").SetData(fc1).CreateSymbol("tanh3");

            // second fullc
            // fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=num_classes)
            Symbol fc2 = new Operator("FullyConnected").SetParam("num_hidden", "10").SetData(tanh3).CreateSymbol("fc2");

            // loss
            // lenet = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
            Symbol lenet = new Operator("SoftmaxOutput").SetData(fc2).SetLabel(data_label).CreateSymbol("softmax");

            System.IO.File.WriteAllText("lenet.json", lenet.ToJSON());

            foreach (var item in lenet.ListAuxiliaryStates())
                Console.WriteLine(item);

            return lenet;
        }

        public void Run()
        {
            Symbol lenet = CreateLenetPython();

            /*setup basic configs*/
            int val_fold = 1;
            int W = 28;
            int H = 28;
            uint batch_size = 256;
            int max_epoch = 20;
            float learning_rate = 0.05f;
            float weight_decay = 0.1f;

            MnistDataSet ds = new MnistDataSet(@"C:\素材\data\train-images.idx3-ubyte", @"C:\素材\data\train-labels.idx1-ubyte");
            //ds.Print();

            List<float> listData = ds.Data;
            List<float> listLabel = ds.Label;
            int dataCount = ds.Count;
            using (FloatListHolder hData = listData.GetHolder())
            using (FloatListHolder hLabel = listLabel.GetHolder())
            {
                NDArray data_array = new NDArray(new Shape((uint)dataCount, 1, (uint)W, (uint)H), ctx_cpu,
                                 false);  // store in main memory, and copy to
                                          // device memory while training

                NDArray label_array = new NDArray(new Shape((uint)dataCount), ctx_cpu,
                    false);  // it's also ok if just store them all in device memory

                data_array.SyncCopyFromCPU(hData.Handle, (ulong)(dataCount * W * H));
                label_array.SyncCopyFromCPU(hLabel.Handle, (ulong)dataCount);
                data_array.WaitToRead();
                label_array.WaitToRead();

                uint train_num = (uint)(dataCount * (1 - val_fold / 10.0));
                train_data = data_array.Slice(0, train_num);
                train_label = label_array.Slice(0, train_num);
                val_data = data_array.Slice(train_num, (uint)dataCount);
                val_label = label_array.Slice(train_num, (uint)dataCount);

                Console.WriteLine("here read fin");

                /*init some of the args*/
                args_map["data"] = data_array.Slice(0, (uint)batch_size).Clone(ctx_dev);
                args_map["data_label"] = label_array.Slice(0, (uint)batch_size).Clone(ctx_dev);
                NDArray.WaitAll();

                Console.WriteLine("here slice  fin");
                lenet.InferArgsMap(ctx_dev, args_map, args_map);
                Optimizer opt = OptimizerRegistry.Find("sgd");
                opt.SetParam("momentum", 0.9)
                   .SetParam("rescale_grad", 1.0)
                   .SetParam("clip_gradient", 10);

                for (int ITER = 0; ITER < max_epoch; ++ITER)
                {
                    uint start_index = 0;
                    while (start_index < train_num)
                    {
                        if (start_index + batch_size > train_num)
                        {
                            start_index = train_num - batch_size;
                        }
                        args_map["data"] =
                            train_data.Slice(start_index, start_index + batch_size)
                                .Clone(ctx_dev);
                        args_map["data_label"] =
                            train_label.Slice(start_index, start_index + batch_size)
                                .Clone(ctx_dev);
                        start_index += batch_size;
                        NDArray.WaitAll();

                        Executor exe = lenet.SimpleBind(ctx_dev, args_map);
                        exe.Forward(true);
                        exe.Backward();
                        exe.UpdateAll(opt, learning_rate, weight_decay);
                        exe.Dispose();
                    }

                    Console.WriteLine("Iter " + ITER + ", accuracy:" + ValAccuracy(batch_size * 10, lenet));
                }
            }
        }

        private float ValAccuracy(uint batch_size, Symbol lenet)
        {
            uint val_num = val_data.GetShape()[0];

            uint correct_count = 0;
            uint all_count = 0;

            uint start_index = 0;
            while (start_index < val_num)
            {
                if (start_index + batch_size > val_num)
                {
                    start_index = val_num - batch_size;
                }
                args_map["data"] =
                    val_data.Slice(start_index, start_index + batch_size).Clone(ctx_dev);
                args_map["data_label"] =
                    val_label.Slice(start_index, start_index + batch_size).Clone(ctx_dev);
                start_index += batch_size;
                NDArray.WaitAll();

                Executor exe = lenet.SimpleBind(ctx_dev, args_map);
                exe.Forward(false);
                var outputs = exe.Outputs;
                NDArray out_cpu = outputs[0].Clone(ctx_cpu);
                NDArray label_cpu =
                    val_label.Slice(start_index - batch_size, start_index).Clone(ctx_cpu);
                NDArray.WaitAll();

                float* dptr_out = out_cpu.GetData();
                float* dptr_label = label_cpu.GetData();
                for (int i = 0; i < batch_size; ++i)
                {
                    float label = dptr_label[i];
                    uint cat_num = out_cpu.GetShape()[1];
                    float p_label = 0, max_p = dptr_out[i * cat_num];
                    for (int j = 0; j < cat_num; ++j)
                    {
                        float p = dptr_out[i * cat_num + j];
                        if (max_p < p)
                        {
                            p_label = j;
                            max_p = p;
                        }
                    }
                    if (label == p_label) correct_count++;
                }
                all_count += batch_size;
                exe.Dispose();
            }
            return (float)(correct_count * 1.0 / all_count);
        }
    }
}
