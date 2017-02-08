using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace MXNetSharp
{
    public class GCHandleManager
    {
        public static GCHandleManager Instance = new GCHandleManager();

        private List<GCHandle> _handles = new List<GCHandle>();

        public void Add(GCHandle handle)
        {
            _handles.Add(handle);
        }

        public void Clear()
        {
            _handles.Clear();
        }
    }

    public unsafe class UnsafeUtils
    {
        public static Dictionary<String, String> CreateStrMap(byte** key, byte** val)
        {
            Dictionary<String, String> map = new Dictionary<string, string>();
            Byte** ppKey = key;
            Byte** ppVal = val;
            while (ppKey != null && ppVal != null)
            {
                String strKey = Marshal.PtrToStringAnsi((IntPtr)(*ppKey));
                String strVal = Marshal.PtrToStringAnsi((IntPtr)(*ppVal));
                if (String.IsNullOrEmpty(strKey) == false && String.IsNullOrEmpty(strVal) == false)
                    map[strKey] = map[strVal];

                ppKey++;
                ppVal++;
            }
            return map;
        }
    }
}
