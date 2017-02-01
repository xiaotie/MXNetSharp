using System;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace MXNetSharp
{
    public unsafe class MemStorge : IDisposable
    {
        private const int MarshalAllocMaxSize = int.MaxValue;

        protected Byte* _handle;
        public Byte* Handle;

        private long _bytes;
        protected long _count;

        public long Bytes
        {
            get { return _bytes; }
        }

        public long Count
        {
            get { return _count; }
        }

        public MemStorge(long bytes)
        {
            _bytes = bytes;
            _count = bytes;
            if (_bytes < MarshalAllocMaxSize)
                _handle =(Byte*)Marshal.AllocHGlobal((int)bytes);
        }

        public virtual void Dispose()
        {
            if(_handle != null)
            {
                if (_bytes < MarshalAllocMaxSize)
                    Marshal.FreeHGlobal((IntPtr)_handle);
            }
        }

        ~MemStorge()
        {
            Dispose();
        }
    }

    public unsafe class VectorF : MemStorge
    {
        public VectorF(long count) : base(count * sizeof(float))
        {
            this._count = count;
        }

        public float* Data
        {
            get { return (float*)_handle; }
        }
    }
}
