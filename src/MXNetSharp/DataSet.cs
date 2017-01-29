using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

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

            LabeledDataSet dataSet = new LabeledDataSet();
            foreach (Byte item in dataBuf)
            {
                float val = item / 256.0f;
                dataSet.Data.Add(val);
            }

            fixed (Byte* pFileBuf = fileBuf)
            {
                int* pLabel = (int*)pFileBuf;
                int count = fileBuf.Length / 4;
                for (int i = 0; i < count; i++)
                {
                    int lebel = ReverseInt(pLabel[i]);
                    dataSet.Label.Add(lebel);
                }
                dataSet.Count = count;
            }
        }
    }
}
