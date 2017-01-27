using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace MXNetSharp.Test
{
    public class NDArrayTest
    {
        public static void TestCreate()
        {
            NDArray nd = new NDArray(new Shape(3, 2, 5), Context.Gpu());
            Console.WriteLine(nd);
        }

        public static void Test()
        {
            TestCreate();
        }
    }
}
