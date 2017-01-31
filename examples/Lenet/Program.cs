using System;
using System.Collections.Generic;
using System.Diagnostics;
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
            Console.ReadKey();
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

        private Symbol CreateLenet()
        {
            Symbol data = Symbol.Variable("data");
            Symbol data_label = Symbol.Variable("data_label");

            // first conv

            //  mx.symbol.Convolution(data = data, kernel = (5, 5), num_filter = 20)
            Symbol conv1 = new Operator("Convolution").SetParam("kernel", "(5, 5)").SetParam("num_filter", "20").SetData(data).CreateSymbol();
            // tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
            Symbol tanh1 = new Operator("Activation").SetParam("act_type", "tanh").SetData(conv1).CreateSymbol();
            // pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max", kernel = (2,2), stride = (2,2))
            Symbol pool1 = new Operator("Pooling").SetParam("pool_type", "max").SetParam("kernel", "(2, 2)").SetParam("stride", "(2, 2)").SetData(tanh1).CreateSymbol();

            // second conv
            // conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=50)
            Symbol conv2 = new Operator("Convolution").SetParam("kernel", "(5, 5)").SetParam("num_filter", "50").SetData(pool1).CreateSymbol();
            // tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
            Symbol tanh2 = new Operator("Activation").SetParam("act_type", "tanh").SetData(conv2).CreateSymbol();
            // pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max", kernel = (2,2), stride = (2,2))
            Symbol pool2 = new Operator("Pooling").SetParam("pool_type", "max").SetParam("kernel", "(2, 2)").SetParam("stride", "(2, 2)").SetData(tanh2).CreateSymbol();

            // first fullc
            // flatten = mx.symbol.Flatten(data=pool2)
            Symbol flatten = new Operator("Flatten").SetData(pool2).CreateSymbol();
            // fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
            Symbol fc1 = new Operator("FullyConnected").SetParam("num_hidden", "500").SetData(flatten).CreateSymbol();
            // tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
            Symbol tanh3 = new Operator("Activation").SetParam("act_type", "tanh").SetData(fc1).CreateSymbol();

            // second fullc
            // fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=num_classes)
            Symbol fc2 = new Operator("FullyConnected").SetParam("num_hidden", "10").SetData(tanh3).CreateSymbol();

            // loss
            // lenet = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
            Symbol lenet = new Operator("SoftmaxOutput").SetData(fc2).SetLabel(data_label).CreateSymbol();

            System.IO.File.WriteAllText("lenet.json", lenet.ToJSON());

            foreach (var item in lenet.ListAuxiliaryStates())
                Console.WriteLine(item);

            return lenet;
        }

        public void Run()
        {
            Symbol lenet = CreateLenet();

            /*setup basic configs*/
            int valFold = 1;
            int W = 28;
            int H = 28;
            uint batchSize = 256;
            int maxEpoch = 20;
            float learning_rate = 0.05f;
            float weight_decay = 0.0001f;

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

                uint train_num = (uint)(dataCount * (1 - valFold / 10.0));
                train_data = data_array.Slice(0, train_num);
                train_label = label_array.Slice(0, train_num);
                val_data = data_array.Slice(train_num, (uint)dataCount);
                val_label = label_array.Slice(train_num, (uint)dataCount);

                Console.WriteLine("Data loaded ok!");

                /*init some of the args*/
                args_map["data"] = data_array.Slice(0, (uint)batchSize).Clone(ctx_dev);
                args_map["data_label"] = label_array.Slice(0, (uint)batchSize).Clone(ctx_dev);
                NDArray.WaitAll();

                Console.WriteLine("Data sliced ok!");
                lenet.InferArgsMap(ctx_dev, args_map, args_map, new XavierInitializer(2));
                Optimizer opt = OptimizerRegistry.Find("sgd");
                opt.SetParam("momentum", 0.9).SetParam("rescale_grad", 1.0 / batchSize);

                for (int ITER = 0; ITER < maxEpoch; ++ITER)
                {
                    Stopwatch sw = new Stopwatch();
                    sw.Start();
                    uint start_index = 0;
                    while (start_index < train_num)
                    {
                        if (start_index + batchSize > train_num)
                        {
                            start_index = train_num - batchSize;
                        }
                        args_map["data"] = train_data.Slice(start_index, start_index + batchSize).Clone(ctx_dev);
                        args_map["data_label"] = train_label.Slice(start_index, start_index + batchSize).Clone(ctx_dev);
                        start_index += batchSize;
                        NDArray.WaitAll();

                        Executor exe = lenet.SimpleBind(ctx_dev, args_map, new XavierInitializer(2));
                        exe.Forward(true);
                        exe.Backward();
                        exe.UpdateAll(opt, learning_rate, weight_decay);
                        exe.Dispose();
                    }
                    sw.Stop();

                    Console.WriteLine("Epoch[" + ITER + "] validation accuracy = " + ValAccuracy(batchSize, lenet) + ", time cost " + sw.Elapsed.TotalSeconds.ToString("0.00") + "s");
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

                Executor exe = lenet.SimpleBind(ctx_dev, args_map, new XavierInitializer(2));
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
