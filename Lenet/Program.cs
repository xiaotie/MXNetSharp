﻿using System;
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

            foreach (String s in lenet.ListArguments())
                Console.WriteLine(s);

            /*setup basic configs*/
            int val_fold = 1;
            int W = 28;
            int H = 28;
            uint batch_size = 42;
            int max_epoch = 100000;
            float learning_rate = 0.0001f;
            float weight_decay = 0.0001f;

            LabeledDataSet ds = new MnistDataSet("","");

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
                Optimizer opt = OptimizerRegistry.Find("ccsgd");
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