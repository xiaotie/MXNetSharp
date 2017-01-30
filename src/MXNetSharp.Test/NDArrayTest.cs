using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Diagnostics;

namespace MXNetSharp.Test
{
    public class NDArrayTest
    {
        public static void TestCreate()
        {
            NDArray ndCPU = new NDArray(new Shape(3, 2, 5), Context.Cpu());
            ndCPU.SetValue(1.0f);
            ndCPU.WaitToWrite();

            Assert.AreEqual(ndCPU.At(1, 1, 1), 1.0f, "NDArray TestSetValue");

            ndCPU.Plus(2.0f);
            ndCPU.WaitToWrite();
            Assert.AreEqual(ndCPU.At(1, 1, 1), 3.0f, "NDArray TestPlus");

            ndCPU.Minus(1.0f);
            ndCPU.WaitToWrite();
            Assert.AreEqual(ndCPU.At(1, 1, 1), 2.0f, "NDArray TestMinus");

            ndCPU.Mul(2.0f);
            ndCPU.WaitToWrite();
            Assert.AreEqual(ndCPU.At(1, 1, 1), 4.0f, "NDArray TestMul");

            ndCPU.Dispose();
        }

        public static void Test()
        {
            TestCreate();
        }
    }
}
