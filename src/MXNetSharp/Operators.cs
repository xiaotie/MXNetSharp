using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace MXNetSharp.Operators
{
    public class Convolution : Operator
    {
        public Shape Kernel { get; set; }
        public Nullable<int> NumFilters { get; set; }

        public Convolution(Shape kernel = null, int numFilters = 1) :base("Convolution")
        {
            this.SetParam("num_filter", numFilters);
            if(kernel != null) this.SetParam("kernel", kernel);
        }

        public void Foo()
        {
            Symbol data = null;

            //python: mx.symbol.Convolution(data = data, kernel = (5, 5), num_filter = 20)

            new Convolution(new Shape(5, 5), 20).CreateSymbol(data);

            new Convolution() { Kernel = new Shape(5, 1), NumFilters = 20 }.CreateSymbol(data);


        }
    }

    public class Activation : Operator
    {
        public enum ActType
        {
            Tanh
        }

        private ActType _type;
        public Activation(ActType type = ActType.Tanh) :base("Activation")
        {
            _type = type;
            String typeStr = null;
            switch(_type)
            {
                case ActType.Tanh:
                default:
                    typeStr = "tanh";
                    break;
            }
            this.SetParam("act_type", typeStr);
        }
    }

    public class Pooling : Operator
    {
        public enum PollingType
        {
            Max
        }

        private PollingType _type;
        public Pooling(PollingType type = PollingType.Max) :base("Pooling")
        {
            _type = type;
            String typeStr = null;
            switch (_type)
            {
                case PollingType.Max:
                default:
                    typeStr = "max";
                    break;
            }
            this.SetParam("pool_type", typeStr);
        }

        public Pooling(Shape kernel = null, Shape stride = null, PollingType type = PollingType.Max) : this(type)
        {
            if (kernel != null) this.SetParam("kernel", kernel);
            if (stride != null) this.SetParam("stride", stride);
        }
    }

    public class Flatten : Operator
    {
        public Flatten() : base("Flatten")
        { }
    }

    public class FullyConnected : Operator
    {

        public FullyConnected(int numHidden):base("FullyConnected")
        {
            this.SetParam("num_hidden", numHidden);
        }
    }

    public class SoftmaxOutput : Operator
    {
        public SoftmaxOutput():base("SoftmaxOutput")
        {
        }
    }
}
