using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Linq;
using System.Threading.Tasks;
using System.Text;

namespace MXNetSharp
{
    /* 参照CAPI.cs */
    using mx_uint = UInt32;
    using mx_float = Single;
    using NDArrayHandle = IntPtr;
    using FunctionHandle = IntPtr;
    using AtomicSymbolCreator = IntPtr;
    using SymbolHandle = IntPtr;
    using AtomicSymbolHandle = IntPtr;
    using ExecutorHandle = IntPtr;
    using DataIterCreator = IntPtr;
    using DataIterHandle = IntPtr;
    using KVStoreHandle = IntPtr;
    using RecordIOHandle = IntPtr;
    using RtcHandle = IntPtr;
    using OptimizerCreator = IntPtr;
    using OptimizerHandle = IntPtr;
    using index_t = UInt32;

    public unsafe class Shape
    {
        /// <summary>
        /// number of dimnsion of the shape
        /// </summary>
        index_t _ndim;

        /// <summary>
        /// in stack space used to store shape when it is small
        /// </summary>
        List<index_t> _dimmensions = new List<index_t>();

        public List<index_t> Data
        {
            get { return _dimmensions; }
        }

        public index_t NDim
        {
            get { return _ndim; }
        }

        public index_t this[int index]
        {
            get { return _dimmensions[index]; }
            set { _dimmensions[index] = value; }
        }

        /// <summary>
        /// total number of elements in the tensor
        /// </summary>
        public uint Size
        {
            get {
                uint size = 1;
                foreach (index_t item in _dimmensions)
                    size *= item;
                return size;
            }
        }

        public Shape()
        {
        }

        /// <summary>
        /// constructor from a vector of index_t
        /// </summary>
        /// <param name="v">the vector</param>
        public Shape(IList<index_t> v)
        {
            _ndim = (uint)v.Count;
            _dimmensions.AddRange(v);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="dimmensions"></param>
        public Shape(params index_t[] dimmensions)
        {
            _ndim = (uint)dimmensions.Length;
            _dimmensions.AddRange(dimmensions);
        }

        public static bool operator ==(Shape a, Shape b)
        {
            if (a._ndim != b._ndim) return false;
            else
            {
                for(int i = 0; i < a._ndim; i++)
                {
                    if (a._dimmensions[i] != b._dimmensions[i]) return false;
                }
                return true;
            }
        }

        public static bool operator !=(Shape a, Shape b)
        {
            return !(a == b);
        }

        public Shape Clone()
        {
            Shape shape = new Shape();
            shape._ndim = this._ndim;
            shape._dimmensions = new List<mx_uint>();
            shape._dimmensions.AddRange(this._dimmensions);
            return shape;
        }

        public Shape Clone(int beginIdx, int endIdx)
        {
            int ndim = endIdx - beginIdx;
            if (ndim <= 1) return new Shape();
            Shape shape = new Shape();
            shape._ndim = (uint)ndim;
            shape._dimmensions = new List<mx_uint>();
            for(int i = 0; i < ndim; ndim++)
            {
                shape._dimmensions.Add(this._dimmensions[beginIdx + i]);
            }
            return shape;
        }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            sb.Append('(');
            for(int i = 0; i < _ndim; i++)
            {
                if (i > 0) sb.Append(',');
                sb.Append(_dimmensions[i]);
            }
            sb.Append(')');
            return sb.ToString();
        }
    };
}
