using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace MXNetSharp.Test
{
    public class OpMapTest
    {
        public static void WriteAllOpNames()
        {
            OpMap map = new OpMap();
            String names = map.GetAllOperatorNames();
            Console.WriteLine("Oprators:" + names);
        }

        public static void Test()
        {
            WriteAllOpNames();
        }
    }
}
