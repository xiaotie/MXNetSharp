using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace MXNetSharp
{
    public class Logging
    {
        public static void CHECK_EQ(int v1, int v2)
        {
            if (v1 != v2)
            {
                String error = CAPI.MXGetLastError();
                if (error != null) Console.WriteLine("Logging CHECK_EQ Failed: " + error);
                else Console.WriteLine("Logging CHECK_EQ Failed");
            }
        }

        public static void CHECK_NE(int v1, int v2)
        {
        }

        public static void LOG_FATAL(String msg)
        {

        }
    }
}
