using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Text;

namespace MXNetSharp
{
    public unsafe class DataSet
    {
        public static int ReverseInt(int i)
        {
            byte* data = stackalloc byte[4];
            *((int*)data) = i;
            byte* dataReverse = stackalloc byte[4];
            dataReverse[0] = data[3];
            dataReverse[1] = data[2];
            dataReverse[2] = data[1];
            dataReverse[3] = data[0];
            return *((int*)dataReverse);
        }
    }

    public unsafe class LabeledDataSet : DataSet
    {
        public int Count;
        public List<float> Data = new List<float>();
        public List<float> Label = new List<float>();
    }

    public unsafe class MnistDataSet : LabeledDataSet
    {
        public MnistDataSet(String fileNameOfDataFile, String fileNameOfLabelFile)
        {
            Byte[] dataBuf = System.IO.File.ReadAllBytes(fileNameOfDataFile);
            Byte[] fileBuf = System.IO.File.ReadAllBytes(fileNameOfLabelFile);

            for(int i = 0; i < dataBuf.Length; i++)
            {
                if (i < 16) continue;
                float val = dataBuf[i] / 256.0f;
                Data.Add(val);
            }

            fixed (Byte* pFileBuf = fileBuf)
            {
                Byte* pLabelData0 = pFileBuf + 8;
                int count = fileBuf.Length - 8;
                for (int i = 0; i < count; i++)
                {
                    Label.Add(pLabelData0[i]);
                }
                Count = count;
            }
        }

        public void Print(int idx = 0, int count = 10)
        {
            if (idx < 0) idx = 0;

            if (count >= this.Count - idx) count = this.Count - idx - 1;

            List<float> data = new List<float>();

            for(int i = 0; i < count; i++)
            {
                int c = (int)Label[i + idx];

                data.Clear();
                int dataIdx = (i + idx) * 28 * 28;
                for(int k = 0; k < 28*28; k++)
                {
                    data.Add(Data[dataIdx + k]);
                }

                PrintImage(c.ToString(), data);

                System.Threading.Thread.Sleep(2000);
            }
        }

        public static void PrintImage(String c, NDArray data)
        {
            PrintImage(c, data.ToList());
        }

        public static void PrintImage(String c, List<float> data)
        {
            Console.WriteLine("============================");
            Console.WriteLine();
            Console.WriteLine(c + ':');
            Console.WriteLine();

            StringBuilder sb = new StringBuilder();
            for (int j = 0; j < data.Count; j++)
            {
                float val = data[j];
                sb.Append(val > 0 ? '#' : '.');
                if (sb.Length == 28)
                {
                    Console.WriteLine(sb.ToString());
                    sb.Clear();
                }
            }
            Console.WriteLine();
        }
    }
}
