using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;

namespace Lenet
{
    using MXNetSharp;
    using MXNetSharp.Operators;

    public class Program
    {
        public static void Main(string[] args)
        {
            LenetModel.RunPredictTest();

            //Lenet lenet = new Lenet();
            //lenet.Run();
            Console.ReadKey();
        }
    }

    public unsafe class LenetModel
    {
        private Symbol _symbol;
        private Executor _exe;
        private NDArray _data;
        private NDArray _label;
        private Context _ctx;
        private NDArray _output;
        private Dictionary<String, NDArray> _args;

        public NDArray Output
        {
            get { return _output; }
        }

        public LenetModel()
        {
            _ctx = Context.Gpu();
            _label = new NDArray(new Shape(1), _ctx);
            _label.SetValue(0);
            _data = new NDArray(new Shape(1, 1, 28, 28), _ctx);
            _data.SetValue(0);
            _output = new NDArray(new Shape(1), Context.Cpu(), false);
            _output.SetValue(0);
            NDArray.WaitAll();
        }

        public void Load(String ndPath)
        {
            Symbol lenet = Lenet.CreateLenet();
            Dictionary<String, NDArray> map = NDArray.LoadToMap(ndPath);
            Dictionary<String, NDArray> args = new Dictionary<string, NDArray>();
            Dictionary<String, NDArray> auxs = new Dictionary<string, NDArray>();
            foreach (var pair in map)
            {
                int idx = pair.Key.IndexOf(':');
                if (idx > 0)
                {
                    String tp = pair.Key.Substring(0, idx);
                    String name = pair.Key.Substring(idx + 1);
                    if (tp == "arg") args[name] = pair.Value;
                    else if (tp == "aux") auxs[name] = pair.Value;
                }
                else
                {
                    args[pair.Key] = pair.Value;
                }
            }

            Context ctx = Context.Gpu();
            args["data"] = _data;
            args["data_label"] = _label;
            lenet.InferArgsMap(ctx, args, args, new XavierInitializer(2));
            _symbol = lenet;
            _args = args;
            _exe = lenet.SimpleBind(_ctx, args, new XavierInitializer(2));
        }

        public String Predict(NDArray nd)
        {
            nd.CopyTo(_data);
            NDArray.WaitAll();
            _exe.Forward(false);
            List<NDArray> outputs = _exe.Outputs;
            NDArray outCpu = outputs[0].Clone(Context.Cpu());
            NDArray.WaitAll();
            float* p = outCpu.GetData();
            int iMax = 0;
            float max = p[0];
            for(int i = 0; i < 10; i ++ )
            {
                if(p[i] > max)
                {
                    max = p[i];
                    iMax = i;
                }
            }
            return iMax.ToString();
        }

        public static void RunPredictTest()
        {
            LenetModel model = new LenetModel();
            model.Load(@"lenet.params");
            MnistDataSet ds = new MnistDataSet(@"C:\素材\data\train-images.idx3-ubyte", @"C:\素材\data\train-labels.idx1-ubyte");
            int W = 28;
            int H = 28;
            List<float> listData = ds.Data;
            List<float> listLabel = ds.Label;
            int dataCount = ds.Count;
            using (FloatListHolder hData = listData.GetHolder())
            using (FloatListHolder hLabel = listLabel.GetHolder())
            {
                NDArray data_array = new NDArray(new Shape((uint)dataCount, 1, (uint)W, (uint)H), Context.Gpu(),
                                 false);  // store in main memory, and copy to
                                          // device memory while training
                NDArray label_array = new NDArray(new Shape((uint)dataCount), Context.Gpu(),
                    false);  // it's also ok if just store them all in device memory

                data_array.SyncCopyFromCPU(hData.Handle, (ulong)(dataCount * W * H));
                label_array.SyncCopyFromCPU(hLabel.Handle, (ulong)dataCount);
                data_array.WaitToRead();
                label_array.WaitToRead();

                for (int i = 0; i < 100; i++)
                {
                    NDArray data = data_array.Slice((uint)i, (uint)i + 1);
                    String output = model.Predict(data);
                    MnistDataSet.PrintImage(output, data);
                    System.Threading.Thread.Sleep(1000);
                }
            }
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

        public static Symbol CreateLenet()
        {
            Symbol data = Symbol.Variable("data");
            Symbol data_label = Symbol.Variable("data_label");

            // first conv

            //  mx.symbol.Convolution(data = data, kernel = (5, 5), num_filter = 20)
            Symbol conv1 = new Convolution(new Shape(5,5), 20).CreateSymbol(data);
            // tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
            Symbol tanh1 = new Activation().CreateSymbol(conv1);
            // pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max", kernel = (2,2), stride = (2,2))
            Symbol pool1 = new Pooling(new Shape(2,2), new Shape(2,2)).CreateSymbol(tanh1);
            // second conv
            // conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=50)
            Symbol conv2 = new Convolution(new Shape(5, 5), 50).CreateSymbol(pool1);
            // tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
            Symbol tanh2 = new Activation().CreateSymbol(conv2);
            // pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max", kernel = (2,2), stride = (2,2))
            Symbol pool2 = new Pooling(new Shape(2, 2), new Shape(2, 2)).CreateSymbol(tanh2);

            // first fullc
            // flatten = mx.symbol.Flatten(data=pool2)
            Symbol flatten = new Flatten().CreateSymbol(pool2);
            // fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
            Symbol fc1 = new FullyConnected(500).CreateSymbol(flatten);
            // tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
            Symbol tanh3 = new Activation().CreateSymbol(fc1);

            // second fullc
            // fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=num_classes)
            Symbol fc2 = new FullyConnected(10).CreateSymbol(tanh3);

            // loss
            // lenet = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
            Symbol lenet = new SoftmaxOutput().CreateSymbol(fc2, data_label);

            System.IO.File.WriteAllText("lenet.json", lenet.ToJSON());

            foreach (var item in lenet.ListAuxiliaryStates())
                Console.WriteLine(item);

            return lenet;
        }

        private Symbol CreateFrom(String jsonFilePath)
        {
            return Symbol.Load(jsonFilePath);
        }

        public void Run()
        {
            Symbol lenet = CreateLenet();

            //Symbol lenet = CreateFrom(@"C:\Works\Projects\80_Project_Python\mxnet\ocr\model\mnist-symbol.json");

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

            NDArray.Save("lenet.params", args_map);
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
