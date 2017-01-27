using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace MXNetSharp.Test
{
    public class Program
    {
        public static void Main(string[] args)
        {
            OpMapTest.Test();
            NDArrayTest.Test();
            Console.WriteLine("Test Finished!");
        }
    }
}
