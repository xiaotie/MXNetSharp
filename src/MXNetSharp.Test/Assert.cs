using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace MXNetSharp.Test
{
    public class Assert
    {
        //public static void AreEqual<T>(T t1, T t2, String testCaseName = "")
        //    where T : struct
        //{
        //    if (t1.Equals(t2) == false)
        //        Console.WriteLine(testCaseName + " failed");
        //    else
        //        Console.WriteLine(testCaseName + " ok");
        //}

        public static void AreEqual(float v1, float v2, String testCaseName = "", float eps = 0.000001f)
        {
            if(Math.Abs(v1-v2) > eps)
                Console.WriteLine(testCaseName + " failed");
            else
                Console.WriteLine(testCaseName + " ok");
        }
    }
}
