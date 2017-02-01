using System;
using System.Collections.Generic;
using System.Text;
using System.Runtime.InteropServices;

namespace MXNetSharp
{
    /* ≤Œ’’CAPI.cs */
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
    using nn_uint = UInt32;
    using OpHandle = IntPtr;
    using GraphHandle = IntPtr;
    using size_t = UInt64;

    public enum DeviceType
    {
        kCPU = 1,
        kGPU = 2,
        kCPUPinned = 3
    }

    public enum ConvolutionCudnnTune
    {
        None = 0,
        fastest = 1,
        limited_workspace = 2,
        off = 3
    };

    /// <summary>
    /// Set layout for input, output and weight. Empty for default layout: NCHW for 2d and NCDHW for 3d.
    /// </summary>
    public enum ConvolutionLayout
    {
        None = 0,
        NCDHW = 1,
        NCHW = 2,
        NDHWC = 3,
        NHWC = 4
    };

    /// <summary>
    /// Activation function to be applied.
    /// </summary>
    public enum ActivationActType
    {
        relu = 0,
        sigmoid = 1,
        softrelu = 2,
        tanh = 3
    };

    /// <summary>
    /// Pooling type to be applied.
    /// </summary>
    public enum PoolingPoolType
    {
        avg = 0,
        max = 1,
        sum = 2
    };

    /// <summary>
    /// Pooling convention to be applied.kValid is default setting of Mxnet
    /// and rounds down the output pooling size.kFull is compatible with
    /// </summary>
    public enum PoolingPoolingConvention
    {
        full = 0,
        valid = 1
    };

    /// <summary>
    /// Softmax Mode. If set to instance, this operator will compute a
    /// softmax for each instance in the batch; this is the default mode. If
    /// set to channel, this operator will compute a num_channel-class
    /// softmax at each position of each instance; this can be used for
    /// </summary>
    public enum SoftmaxActivationMode
    {
        channel = 0,
        instance = 1
    };

    /// <summary>
    /// If set to null, op will do nothing on output gradient.If set to
    /// batch, op will normalize gradient by divide batch sizeIf set to
    /// </summary>
    public enum SoftmaxOutputNormalization
    {
        batch = 0,
        none = 1,
        valid = 2
    };

    #region utils classes

    public unsafe class StringHolder : IDisposable
    {
        public String String
        {
            get; private set;
        }

        private Byte* _handle;
        public Byte* Handle
        {
            get
            {
                return _handle;
            }
        }

        public StringHolder(String str)
        {
            String = str;
            _handle = (Byte*)Marshal.StringToHGlobalAnsi(str);
        }

        public StringHolder(byte* chars)
        {
            String = Marshal.PtrToStringAnsi((IntPtr)chars);
            _handle = null;
        }

        ~StringHolder()
        {
            Dispose();
        }

        public void Dispose()
        {
            if (_handle != null)
            {
                Marshal.FreeHGlobal((IntPtr)_handle);
                _handle = null;
            }
        }
    }

    public unsafe class UInt32ListHolder : IDisposable
    {
        private UInt32* _handle;
        public UInt32* Handle
        {
            get
            {
                return _handle;
            }
        }

        public UInt32ListHolder(IList<UInt32> list)
        {
            _handle = (UInt32*)Marshal.AllocHGlobal(sizeof(UInt32) * list.Count);
            for (int i = 0; i < list.Count; i++)
                _handle[i] = list[i];
        }

        ~UInt32ListHolder()
        {
            Dispose();
        }

        public void Dispose()
        {
            if (_handle != null)
            {
                Marshal.FreeHGlobal((IntPtr)_handle);
                _handle = null;
            }
        }
    }

    public unsafe class FloatListHolder : IDisposable
    {
        private float* _handle;
        public float* Handle
        {
            get
            {
                return _handle;
            }
        }

        public FloatListHolder(IList<float> list)
        {
            _handle = (float*)Marshal.AllocHGlobal(sizeof(float) * list.Count);
            for (int i = 0; i < list.Count; i++)
                _handle[i] = list[i];
        }

        ~FloatListHolder()
        {
            Dispose();
        }

        public void Dispose()
        {
            if (_handle != null)
            {
                Marshal.FreeHGlobal((IntPtr)_handle);
                _handle = null;
            }
        }
    }

    public unsafe class IntPtrListHolder : IDisposable
    {
        private IntPtr* _handle;
        public IntPtr* Handle
        {
            get
            {
                return _handle;
            }
        }

        public IntPtrListHolder(IList<IntPtr> list)
        {
            _handle = (IntPtr*)Marshal.AllocHGlobal(sizeof(IntPtr) * list.Count);
            for (int i = 0; i < list.Count; i++)
                _handle[i] = list[i];
        }

        ~IntPtrListHolder()
        {
            Dispose();
        }

        public void Dispose()
        {
            if (_handle != null)
            {
                Marshal.FreeHGlobal((IntPtr)_handle);
                _handle = null;
            }
        }
    }

    public unsafe class StringListHolder : IDisposable
    {
        private IList<String> _list;
        private List<StringHolder> _hList;

        private Byte** _pointer;

        public Byte** Pointer
        {
            get { return _pointer; }
        }

        public StringListHolder(IList<String> list)
        {
            _list = list;
            _hList = new List<StringHolder>(list.Count);
            _pointer = (Byte**)Marshal.AllocHGlobal(sizeof(IntPtr) * list.Count);
            for (int i = 0; i < list.Count; i++)
            {
                StringHolder h = new StringHolder(list[i]);
                _hList.Add(h);
                _pointer[i] = h.Handle;
            }
        }

        ~StringListHolder()
        {
            Dispose();
        }

        public void Dispose()
        {
            if (_pointer == null) return;

            Marshal.FreeHGlobal((IntPtr)_pointer);
            _pointer = null;

            if (_hList != null)
            {
                foreach (var item in _hList)
                    item.Dispose();

                _hList = null;
            }
        }
    }

    public static class ClassHelper
    {
        public static NDArrayHandle[] GetHandles(this List<NDArray> list)
        {
            NDArrayHandle[] handles = new NDArrayHandle[list.Count];
            for (int i = 0; i < list.Count; i++)
                handles[i] = list[i].Handle;
            return handles;
        }

        public static NDArrayHandle[] GetHandles(this IEnumerable<NDArray> collection)
        {
            List<NDArray> list = new List<NDArray>();
            list.AddRange(collection);
            return list.GetHandles();
        }

        public static SymbolHandle[] GetHandles(this List<Symbol> list)
        {
            SymbolHandle[] handles = new SymbolHandle[list.Count];
            for (int i = 0; i < list.Count; i++)
                handles[i] = list[i].Handle;
            return handles;
        }

        public static mx_uint[] GetValues(this List<OpReqType> list)
        {
            mx_uint[] data = new mx_uint[list.Count];
            for (int i = 0; i < list.Count; i++)
                data[i] = (mx_uint)list[i];
            return data;
        }

        public static StringListHolder GetHolder(this IList<String> list)
        {
            return new StringListHolder(list);
        }

        public static StringListHolder GetHolder(this IEnumerable<String> collection)
        {
            List<String> list = new List<string>();
            list.AddRange(collection);
            return new StringListHolder(list);
        }

        public static IntPtrListHolder GetHolder(this IList<IntPtr> list)
        {
            return new IntPtrListHolder(list);
        }

        public static UInt32ListHolder GetHolder(this IList<UInt32> list)
        {
            return new UInt32ListHolder(list);
        }

        public static FloatListHolder GetHolder(this IList<float> list)
        {
            return new FloatListHolder(list);
        }
    }

    #endregion

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
            get
            {
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
            for (int i = 0; i < ndim; ndim++)
            {
                shape._dimmensions.Add(this._dimmensions[beginIdx + i]);
            }
            return shape;
        }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            sb.Append('(');
            for (int i = 0; i < _ndim; i++)
            {
                if (i > 0) sb.Append(',');
                sb.Append(_dimmensions[i]);
            }
            sb.Append(')');
            return sb.ToString();
        }
    };

    /// <summary>
    /// Context interface
    /// </summary>
    public class Context
    {
        private DeviceType _type;

        private int _id;

        /// <summary>
        /// the type of the device
        /// </summary>
        public DeviceType DeviceType
        {
            get { return _type; }
        }

        /// <summary>
        /// the id of the device
        /// </summary>
        public int DeviceId
        {
            get { return _id; }
        }

        /// <summary>
        /// Context constructor
        /// </summary>
        /// <param name="type">type of the device</param>
        /// <param name="id">id of the device</param>
        public Context(DeviceType type, int id)
        {
            this._type = type;
            this._id = id;
        }

        /// <summary>
        /// Return a GPU context
        /// </summary>
        /// <param name="device_id">id of the device</param>
        /// <returns>the corresponding GPU context</returns>
        public static Context Gpu(int device_id = 0)
        {
            return new Context(DeviceType.kGPU, device_id);
        }

        /// <summary>
        /// Return a CPU context
        /// </summary>
        /// <param name="device_id">id of the device. this is not needed by CPU</param>
        /// <returns>the corresponding CPU context</returns>
        public static Context Cpu(int device_id = 0)
        {
            return new Context(DeviceType.kCPU, device_id);
        }
    }

    /// <summary>
    /// Class to store NDArrayHandle
    /// </summary>
    public class NDBlob : IDisposable
    {
        private NDArrayHandle _handle;

        public NDArrayHandle Handle
        {
            get { return _handle; }
        }

        /// <summary>
        /// construct with a NDArrayHandle
        /// </summary>
        /// <param name="handle">handle NDArrayHandle to store</param>
        public NDBlob(NDArrayHandle handle)
        {
            _handle = handle;
        }

        #region IDisposable Support
        private bool disposedValue = false; // To detect redundant calls

        ~NDBlob()
        {
            Dispose();
        }

        public void Dispose()
        {
            if (!disposedValue)
            {
                CAPI.MXNDArrayFree(_handle);
                _handle = IntPtr.Zero;

                disposedValue = true;
            }
        }
        #endregion
    }

    public unsafe class NDArray : IDisposable
    {
        private NDBlob _blob;

        public NDArrayHandle Handle
        {
            get { return _blob.Handle; }
        }

        public size_t Size
        {
            get
            {
                size_t ret = 1;
                foreach (var item in this.GetShape())
                {
                    ret *= item;
                }
                return ret;
            }
        }

        public NDArray()
        {
            NDArrayHandle handle;
            Logging.CHECK_EQ(CAPI.MXNDArrayCreateNone(&handle), 0);
            _blob = new NDBlob(handle);
        }

        public NDArray(NDArrayHandle handle)
        {
            _blob = new NDBlob(handle);
        }

        /// <summary>
        /// construct a new dynamic NDArray
        /// </summary>
        /// <param name="shape">the shape of array</param>
        /// <param name="context">context of NDArray</param>
        /// <param name="delay_alloc">whether delay the allocation</param>
        public NDArray(List<mx_uint> shape, Context context, bool delay_alloc = true)
        {
            mx_uint[] arr = shape.ToArray();
            fixed (mx_uint* pArr = arr)
            {
                NDArrayHandle handle;
                Logging.CHECK_EQ(CAPI.MXNDArrayCreate(pArr, (uint)shape.Count, (int)context.DeviceType,
                                         context.DeviceId, delay_alloc ? 1 : 0, &handle),
                         0);
                _blob = new NDBlob(handle);
            }
        }

        public NDArray(Shape shape, Context context, bool delay_alloc = true)
            : this(shape.Data, context, delay_alloc)
        {
        }

        public NDArray(mx_float* data, uint size)
        {
            NDArrayHandle handle;
            Logging.CHECK_EQ(CAPI.MXNDArrayCreateNone(&handle), 0);
            CAPI.MXNDArraySyncCopyFromCPU(handle, data, size);
            _blob = new NDBlob(handle);
        }

        public NDArray(List<mx_float> data)
        {
            mx_float[] dataArr = data.ToArray();
            fixed (mx_float* pDataArr = dataArr)
            {
                NDArrayHandle handle;
                Logging.CHECK_EQ(CAPI.MXNDArrayCreateNone(&handle), 0);
                CAPI.MXNDArraySyncCopyFromCPU(handle, pDataArr, (uint)data.Count);
                _blob = new NDBlob(handle);
            }
        }

        public NDArray(List<mx_float> data, Shape shape, Context context)
        {
            mx_float[] dataArr = data.ToArray();
            mx_uint[] arr = shape.Data.ToArray();
            fixed (mx_uint* pArr = arr)
            fixed (mx_float* pDataArr = dataArr)
            {
                NDArrayHandle handle;
                Logging.CHECK_EQ(CAPI.MXNDArrayCreate(pArr, (uint)shape.NDim, (int)context.DeviceType,
                                         context.DeviceId, 0, &handle),
                         0);
                CAPI.MXNDArraySyncCopyFromCPU(handle, pDataArr, shape.Size);
                _blob = new NDBlob(handle);
            }
        }

        public NDArray(mx_float* data, Shape shape, Context context)
        {

            mx_uint[] arr = shape.Data.ToArray();
            fixed (mx_uint* pArr = arr)
            {
                NDArrayHandle handle;
                Logging.CHECK_EQ(CAPI.MXNDArrayCreate(pArr, (uint)shape.NDim, (int)context.DeviceType,
                                         context.DeviceId, 0, &handle),
                         0);
                CAPI.MXNDArraySyncCopyFromCPU(handle, data, shape.Size);
                _blob = new NDBlob(handle);
            }
        }

        #region operators

        public static NDArray operator +(NDArray lhs, mx_float scalar)
        {
            NDArray ret = new NDArray();
            new Operator("_plus_scalar").Set(lhs, scalar).Invoke(ret);
            return ret;
        }

        public static NDArray operator -(NDArray lhs, mx_float scalar)
        {
            NDArray ret = new NDArray();
            new Operator("_minus_scalar").Set(lhs, scalar).Invoke(ret);
            return ret;
        }

        public static NDArray operator *(NDArray lhs, mx_float scalar)
        {
            NDArray ret = new NDArray();
            new Operator("_mul_scalar").Set(lhs, scalar).Invoke(ret);
            return ret;
        }

        public static NDArray operator /(NDArray lhs, mx_float scalar)
        {
            NDArray ret = new NDArray();
            new Operator("_div_scalar").Set(lhs, scalar).Invoke(ret);
            return ret;
        }

        public static NDArray operator +(NDArray lhs, NDArray rhs)
        {
            NDArray ret = new NDArray();
            new Operator("_plus").Set(lhs, rhs).Invoke(ret);
            return ret;
        }

        public static NDArray operator -(NDArray lhs, NDArray rhs)
        {
            NDArray ret = new NDArray();
            new Operator("_minus").Set(lhs, rhs).Invoke(ret);
            return ret;
        }

        public static NDArray operator *(NDArray lhs, NDArray rhs)
        {
            NDArray ret = new NDArray();
            new Operator("_mul").Set(lhs, rhs).Invoke(ret);
            return ret;
        }

        public static NDArray operator /(NDArray lhs, NDArray rhs)
        {
            NDArray ret = new NDArray();
            new Operator("_div").Set(lhs, rhs).Invoke(ret);
            return ret;
        }

        public void SetValue(mx_float scalar)
        {
            new Operator("_set_value").Set(scalar).Invoke(this);
        }

        public void Plus(mx_float scalar)
        {
            new Operator("_plus_scalar").Set(this, scalar).Invoke(this);
        }

        public void Minus(mx_float scalar)
        {
            new Operator("_minus_scalar").Set(this, scalar).Invoke(this);
        }

        public void Mul(mx_float scalar)
        {
            new Operator("_mul_scalar").Set(this, scalar).Invoke(this);
        }

        public void Div(mx_float scalar)
        {
            new Operator("_div_scalar").Set(this, scalar).Invoke(this);
        }

        public void Plus(NDArray nd)
        {
            new Operator("_plus_scalar").Set(this, nd).Invoke(this);
        }

        public void Minus(NDArray nd)
        {
            new Operator("_minus_scalar").Set(this, nd).Invoke(this);
        }

        public void Mul(NDArray nd)
        {
            new Operator("_mul_scalar").Set(this, nd).Invoke(this);
        }

        public void Div(NDArray nd)
        {
            new Operator("_div_scalar").Set(this, nd).Invoke(this);
        }

        public NDArray ArgmaxChannel()
        {
            NDArray ret = new NDArray();
            new Operator("argmax_channel").Set(this).Invoke(ret);
            return ret;
        }

        #endregion

        /// <summary>
        /// 
        /// </summary>
        /// <returns>the shape of current NDArray, in the form of mx_uint vector</returns>
        public List<mx_uint> GetShape()
        {
            mx_uint* out_pdata;
            mx_uint out_dim;
            CAPI.MXNDArrayGetShape(Handle, &out_dim, &out_pdata);
            List<mx_uint> ret = new List<mx_uint>();
            for (mx_uint i = 0; i < out_dim; ++i)
            {
                ret.Add(out_pdata[i]);
            }
            return ret;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns>the data pointer to the current NDArray</returns>
        public mx_float* GetData()
        {
            mx_float* ret;
            Logging.CHECK_NE((int)(GetContext().DeviceType), (int)DeviceType.kGPU);
            CAPI.MXNDArrayGetData(_blob.Handle, &ret);
            return ret;
        }

        /// <summary>
        /// get the context of NDArray
        /// </summary>
        /// <returns>the context of NDArray</returns>
        public Context GetContext()
        {
            int out_dev_type;
            int out_dev_id;
            CAPI.MXNDArrayGetContext(_blob.Handle, &out_dev_type, &out_dev_id);
            return new Context((DeviceType)out_dev_type, out_dev_id);
        }

        /// <summary>
        /// Do a synchronize copy from a continugous CPU memory region.
        /// This function will call WaitToWrite before the copy is performed.
        /// This is useful to copy data from existing memory region that are
        /// not wrapped by NDArray(thus dependency not being tracked).
        /// </summary>
        /// <param name="data">the data source to copy from.</param>
        /// <param name="size">the memory size we want to copy from.</param>
        public void SyncCopyFromCPU(mx_float* data, size_t size)
        {
            CAPI.MXNDArraySyncCopyFromCPU(_blob.Handle, data, size);
        }

        /// <summary>
        /// Do a synchronize copy to a continugous CPU memory region.
        /// This function will call WaitToRead before the copy is performed.
        /// This is useful to copy data from existing memory region that are
        /// not wrapped by NDArray(thus dependency not being tracked).
        /// </summary>
        /// <param name="data">the data source to copyinto.</param>
        /// <param name="size">the memory size we want to copy into. Defualt value is Size()</param>
        public void SyncCopyToCPU(mx_float* data, size_t size)
        {
            CAPI.MXNDArraySyncCopyToCPU(_blob.Handle, data, size > 0 ? size : Size);
        }

        /// <summary>
        /// Copy the content of current array to other.
        /// </summary>
        /// <param name="other">other the new context of this NDArray</param>
        public void CopyTo(NDArray other)
        {
            new Operator("_copyto").Set(this).Invoke(other);
        }

        /// <summary>
        /// return a new copy this NDArray
        /// </summary>
        /// <param name="context"></param>
        /// <returns></returns>
        public NDArray Clone(Context context)
        {
            NDArray ret = new NDArray(GetShape(), context);
            new Operator("_copyto").Set(this).Invoke(ret);
            return ret;
        }

        /// <summary>
        /// return offset of the element at (h, w)
        /// </summary>
        /// <param name="h">height position</param>
        /// <param name="w">width position</param>
        /// <returns>offset of two dimensions array</returns>
        public size_t Offset(size_t h = 0, size_t w = 0)
        {
            return (h * GetShape()[1]) + w;
        }

        /// <summary>
        /// return offset of three dimensions array
        /// </summary>
        /// <param name="c">channel position</param>
        /// <param name="h">height position</param>
        /// <param name="w">width position</param>
        /// <returns>offset of three dimensions array</returns>
        public size_t Offset(size_t c, size_t h, size_t w)
        {
            var shape = GetShape();
            return h * shape[0] * shape[2] + w * shape[0] + c;
        }

        /// <summary>
        /// return value of the element at (h, w)
        /// </summary>
        /// <param name="h">height position</param>
        /// <param name="w">width position</param>
        /// <returns>value of two dimensions array</returns>
        public mx_float At(size_t h, size_t w)
        {
            return GetData()[Offset(h, w)];
        }

        /// <summary>
        /// return value of three dimensions array
        /// </summary>
        /// <param name="c">channel position</param>
        /// <param name="h">height position</param>
        /// <param name="w">width position</param>
        /// <returns>value of three dimensions array</returns>
        public mx_float At(size_t c, size_t h, size_t w)
        {
            return GetData()[Offset(c, h, w)];
        }

        public mx_float this[int h, int w]
        {
            get { return GetData()[Offset((uint)h,(uint)w)]; }
        }

        public mx_float this[int c, int h, int w]
        {
            get { return GetData()[Offset((uint)c, (uint)h, (uint)w)]; }
        }

        /// <summary>
        /// Slice a NDArray
        /// </summary>
        /// <param name="begin">begin index in first dim</param>
        /// <param name="end">end index in first dim</param>
        /// <returns>sliced NDArray</returns>
        public NDArray Slice(mx_uint begin, mx_uint end)
        {
            NDArrayHandle handle;
            Logging.CHECK_EQ(CAPI.MXNDArraySlice(Handle, begin, end, &handle), 0);
            return new NDArray(handle);
        }

        /// <summary>
        /// Return a reshaped NDArray that shares memory with current one
        /// </summary>
        /// <param name="newShape">the new shape</param>
        /// <returns>reshaped NDarray</returns>
        public NDArray Reshape(Shape newShape)
        {
            NDArrayHandle handle;
            int[] dims = new int[newShape.NDim];
            for (int i = 0; i < newShape.NDim; i++)
                dims[i] = (int)newShape[i];
            fixed (int* pDim = dims)
            {
                Logging.CHECK_EQ(
               CAPI.MXNDArrayReshape(Handle, (int)newShape.NDim, pDim, &handle), 0);
            }
            return new NDArray(handle);
        }

        /// <summary>
        /// Block until all the pending write operations with respect
        /// to current NDArray are finished, and read can be performed.
        /// </summary>
        public void WaitToRead()
        {
            Logging.CHECK_EQ(CAPI.MXNDArrayWaitToRead(_blob.Handle), 0);
        }

        /// <summary>
        /// Block until all the pending read/write operations with respect
        /// to current NDArray are finished, and read/write can be performed.
        /// </summary>
        public void WaitToWrite()
        {
            Logging.CHECK_EQ(CAPI.MXNDArrayWaitToWrite(_blob.Handle), 0);
        }

        /// <summary>
        /// Block until all the pending read/write operations with respect
        /// to current NDArray are finished, and read/write can be performed.
        /// </summary>
        public static void WaitAll()
        {
            Logging.CHECK_EQ(CAPI.MXNDArrayWaitAll(), 0);
        }

        /// <summary>
        /// Sample gaussian distribution for each elements of out.
        /// </summary>
        /// <param name="mu">mean of gaussian distribution.</param>
        /// <param name="sigma">standard deviation of gaussian distribution.</param>
        /// <param name="pOut">output NDArray</param>
        public static void SampleGaussian(mx_float mu, mx_float sigma, NDArray pOut)
        {
            new Operator("_sample_normal").Set(mu, sigma).Invoke(pOut);
        }

        /// <summary>
        /// Sample uniform distribution for each elements of out.
        /// </summary>
        /// <param name="begin">lower bound of distribution.</param>
        /// <param name="end">upper bound of distribution.</param>
        /// <param name="ndOut">output NDArray.</param>
        public static void SampleUniform(mx_float begin, mx_float end, NDArray ndOut)
        {
            new Operator("_sample_uniform").Set(begin, end).Invoke(ndOut);
        }

        #region save & load

        /// <summary>
        /// Load NDArrays from binary file.
        /// </summary>
        /// <param name="file_name">name of the binary file</param>
        /// <param name="array_list">a list of NDArrays returned, do not fill the list if nullptr is given.</param>
        /// <param name="array_map">
        /// a map from names to NDArrays returned, do not fill the map 
        /// if nullptr is given or no names is stored in binary file.
        /// </param>
        public static void Load(String file_name,
                   List<NDArray> array_list = null,
                   Dictionary<String, NDArray> array_map = null)
        {
            mx_uint out_size, out_name_size;
            NDArrayHandle* out_arr;
            byte** out_names;
            Logging.CHECK_EQ(CAPI.MXNDArrayLoad(file_name, &out_size, &out_arr, &out_name_size,
                                   &out_names),
                     0);
            if (array_list != null)
            {
                for (mx_uint i = 0; i < out_size; ++i)
                {
                    array_list.Add(new NDArray(out_arr[i]));
                }
            }

            if (array_map != null && out_name_size > 0)
            {
                Logging.CHECK_EQ((int)out_name_size, (int)out_size);
                for (mx_uint i = 0; i < out_size; ++i)
                {
                    array_map[Marshal.PtrToStringAnsi((IntPtr)(out_names[i]))] = new NDArray(out_arr[i]);
                }
            }
        }

        public List<float> ToList()
        {
            NDArray nd = this;
            Context ctx = this.GetContext();
            if (ctx.DeviceType != DeviceType.kCPU)
                nd = this.Clone(Context.Cpu());
            NDArray.WaitAll();
            float* pData = nd.GetData();
            int count = (int)nd.Size;
            List<float> list = new List<float>(count);
            for (int i = 0; i < count; i++)
                list.Add(pData[i]);

            if (nd != this) nd.Dispose();

            return list;
        }

        /// <summary>
        /// Load map of NDArrays from binary file.
        /// </summary>
        /// <param name="file_name">name of the binary file.</param>
        /// <returns>a map from names to NDArrays.</returns>
        public static Dictionary<String, NDArray> LoadToMap(String file_name)
        {
            Dictionary<String, NDArray> map = new Dictionary<String, NDArray>();
            Load(file_name, null, map);
            return map;
        }

        /// <summary>
        /// Load list of NDArrays from binary file.
        /// </summary>
        /// <param name="file_name">name of the binary file.</param>
        /// <returns>a list of NDArrays.</returns>
        public static List<NDArray> LoadToList(String file_name)
        {
            List<NDArray> list = new List<NDArray>();
            Load(file_name, list, null);
            return list;
        }

        /// <summary>
        /// save a map of string->NDArray to binary file.
        /// </summary>
        /// <param name="file_name">name of the binary file.</param>
        /// <param name="array_map">a map from names to NDArrays.</param>
        public static void Save(String file_name,
                   Dictionary<String, NDArray> array_map)
        {
            using (StringListHolder hKeys = array_map.Keys.GetHolder())
            {
                NDArrayHandle[] handles = array_map.Values.GetHandles();
                fixed (NDArrayHandle* pHandle = handles)
                {
                    Logging.CHECK_EQ(CAPI.MXNDArraySave(file_name, (uint)array_map.Count, pHandle, hKeys.Pointer), 0);
                }
            }
        }

        /// <summary>
        /// save a list of NDArrays to binary file.
        /// </summary>
        /// <param name="file_name">name of the binary file.</param>
        /// <param name="arrayList">a list of NDArrays.</param>
        public static void Save(String file_name, List<NDArray> arrayList)
        {
            NDArrayHandle[] handles = arrayList.GetHandles();
            fixed (NDArrayHandle* pHandle = handles)
            {
                Logging.CHECK_EQ(CAPI.MXNDArraySave(file_name, (uint)arrayList.Count, pHandle, null), 0);
            }
        }

        public void Save(String fileName)
        {
            List<NDArray> list = new List<NDArray>();
            NDArray.Save(fileName, list);
        }

        public static NDArray Load(String fileName)
        {
            List<NDArray> list = NDArray.LoadToList(fileName);
            return list[0];
        }

        #endregion

        #region IDisposable Support
        private bool disposedValue = false; // To detect redundant calls

        ~NDArray()
        {
            Dispose();
        }

        public void Dispose()
        {
            if (!disposedValue)
            {
                disposedValue = true;
                if (_blob != null)
                {
                    _blob.Dispose();
                    _blob = null;
                }
            }
        }
        #endregion
    }

    /// <summary>
    /// OpMap instance holds a map of all the symbol creators so we can
    /// get symbol creators by name.
    /// 
    /// This is used internally by Symbol and Operator.
    /// </summary>
    public unsafe class OpMap
    {
        Dictionary<String, AtomicSymbolCreator> symbol_creators_ = new Dictionary<String, AtomicSymbolCreator>();
        Dictionary<String, OpHandle> op_handles_ = new Dictionary<string, OpHandle>();

        public OpMap()
        {
            mx_uint num_symbol_creators = 0;
            AtomicSymbolCreator* symbol_creators = null;
            int r = CAPI.MXSymbolListAtomicSymbolCreators(&num_symbol_creators, &symbol_creators);
            Logging.CHECK_EQ(r, 0);
            for (mx_uint i = 0; i < num_symbol_creators; i++)
            {
                byte* name;
                byte* description;
                mx_uint num_args;
                byte** arg_names;
                byte** arg_type_infos;
                byte** arg_descriptions;
                byte* key_var_num_args;
                r = CAPI.MXSymbolGetAtomicSymbolInfo(symbol_creators[i], &name, &description,
                  &num_args, &arg_names, &arg_type_infos,
                  &arg_descriptions, &key_var_num_args);
                Logging.CHECK_EQ(r, 0);
                String sName = Marshal.PtrToStringAnsi((IntPtr)name);
                symbol_creators_[sName] = symbol_creators[i];
            }

            nn_uint num_ops;
            byte** op_names;
            r = CAPI.NNListAllOpNames(&num_ops, &op_names);
            Logging.CHECK_EQ(r, 0);
            for (nn_uint i = 0; i < num_ops; i++)
            {
                OpHandle handle;
                r = CAPI.NNGetOpHandle(op_names[i], &handle);
                Logging.CHECK_EQ(r, 0);
                String sName = Marshal.PtrToStringAnsi((IntPtr)(op_names[i]));
                op_handles_[sName] = handle;
            }
        }

        public String GetAllOperatorNames()
        {
            StringBuilder sb = new StringBuilder();
            var keys = op_handles_.Keys;
            int idx = 0;
            foreach (var key in keys)
            {
                if (idx > 0) sb.Append(',');
                sb.Append(key);
                idx++;
            }
            return sb.ToString();
        }

        public OpHandle GetOpHandle(String name)
        {
            return op_handles_[name];
        }

        public AtomicSymbolCreator GetSymbolCreator(String name)
        {
            if (symbol_creators_.ContainsKey(name) == false)
                return GetOpHandle(name);
            return symbol_creators_[name];
        }
    }

    /// <summary>
    /// Operator interface
    /// </summary>
    public unsafe class Operator
    {
        private static OpMap op_map_ = new OpMap();

        Dictionary<String, String> params_desc_ = new Dictionary<string, string>();
        bool variable_params_ = false;
        Dictionary<String, String> params_ = new Dictionary<string, string>();
        List<SymbolHandle> input_symbols = new List<SymbolHandle>();
        List<NDArrayHandle> input_ndarrays = new List<NDArrayHandle>();
        List<String> input_keys = new List<String>();
        List<String> arg_names_ = new List<String>();
        AtomicSymbolCreator handle_;

        private static Dictionary<String, int> _symbolNames = new Dictionary<string, int>();

        private static String GetSymbolName(String opName)
        {
            int num = 0;
            if (_symbolNames.ContainsKey(opName) == false) _symbolNames[opName] = num;
            else
            {
                num = _symbolNames[opName] + 1;
                _symbolNames[opName] = num;
            }
            return opName + "_" + num;
        }

        public Operator SetParam(int pos, NDArray val)
        {
            input_ndarrays.Add(val.Handle);
            return this;
        }

        public Operator SetParam(int pos, Symbol val)
        {
            input_symbols.Add(val.Handle);
            return this;
        }

        /// <summary>
        /// set config parameters
        /// </summary>
        /// <param name="name">name of the config parameter</param>
        /// <param name="val">value of the config parameter</param>
        /// <returns>reference of self</returns>
        public Operator SetParam(String name, Object val)
        {
            params_[name] = val.ToString();
            return this;
        }

        public Operator SetParam(String name, bool val)
        {
            params_[name] = val ? "1" : "0";
            return this;
        }

        public Operator SetData(Symbol data)
        {
            return SetInput("data", data);
        }

        public Operator SetLabel(Symbol label)
        {
            return SetInput("label", label);
        }

        /// <summary>
        /// set config parameters from positional inputs
        /// </summary>
        /// <param name="pos">the position of parameter</param>
        /// <param name="val">value of the config parameter</param>
        /// <returns>reference of self</returns>
        public Operator SetParam(int pos, Object val)
        {
            params_[arg_names_[pos]] = val.ToString();
            return this;
        }

        /// <summary>
        /// add an input symbol
        /// </summary>
        /// <param name="name">name of the input symbol</param>
        /// <param name="symbol">the input symbol</param>
        /// <returns>reference of self</returns>
        public Operator SetInput(String name, Symbol symbol)
        {
            input_keys.Add(name);
            input_symbols.Add(symbol.Handle);
            return this;
        }

        /// <summary>
        /// add input symbols
        /// </summary>
        /// <param name="symbol">the input symbol</param>
        /// <returns>reference of self</returns>
        public Operator Add(Symbol symbol)
        {
            input_symbols.Add(symbol.Handle);
            return this;
        }

        /// <summary>
        /// add a list of input symbols
        /// </summary>
        /// <param name="symbols">the vector of the input symbols</param>
        /// <returns>reference of self</returns>
        public Operator AddRange(List<Symbol> symbols)
        {
            foreach (Symbol item in symbols)
                input_symbols.Add(item.Handle);
            return this;
        }

        /// <summary>
        /// add an input ndarray
        /// </summary>
        /// <param name="name">name of the input ndarray</param>
        /// <param name="ndarray">the input ndarray</param>
        /// <returns>reference of self</returns>
        public Operator SetInput(String name, NDArray ndarray)
        {
            input_keys.Add(name);
            input_ndarrays.Add(ndarray.Handle);
            return this;
        }

        /// <summary>
        /// add an input ndarray
        /// </summary>
        /// <param name="ndarray">the input ndarray</param>
        public void PushInput(NDArray ndarray)
        {
            input_ndarrays.Add(ndarray.Handle);
        }

        /// <summary>
        /// add input ndarrays
        /// </summary>
        /// <param name="ndarray">the input ndarray</param>
        /// <returns>reference of self</returns>
        public Operator Add(NDArray ndarray)
        {
            input_ndarrays.Add(ndarray.Handle);
            return this;
        }

        /// <summary>
        /// add a list of input ndarrays
        /// </summary>
        /// <param name="ndarrays">the vector of the input ndarrays</param>
        /// <returns>reference of self</returns>
        public Operator AddRange(List<NDArray> ndarrays)
        {
            foreach (NDArray item in ndarrays)
                input_ndarrays.Add(item.Handle);
            return this;
        }

        public Operator Set(params Object[] args)
        {
            for (int i = 0; i < args.Length; i++)
            {
                Object arg = args[i];
                if (arg is Symbol)
                    SetParam(i, (Symbol)arg);
                else if (arg is NDArray)
                    SetParam(i, (NDArray)arg);
                else
                    SetParam(i, arg);
            }
            return this;
        }

        private String _opName;

        /// <summary>
        /// Operator constructor
        /// </summary>
        /// <param name="name">type of the operator</param>
        public Operator(String operatorName)
        {
            _opName = operatorName;
            handle_ = op_map_.GetSymbolCreator(operatorName);
            Byte* name;
            Byte* description;
            mx_uint num_args;
            Byte** arg_names;
            Byte** arg_type_infos;
            Byte** arg_descriptions;
            Byte* key_var_num_args;
            Logging.CHECK_EQ(CAPI.MXSymbolGetAtomicSymbolInfo(handle_,
                &name,
                &description,
                &num_args,
                &arg_names,
                &arg_type_infos,
                &arg_descriptions,
                &key_var_num_args),0);
            for (mx_uint i = 0; i < num_args; ++i)
            {
                byte* pArgName = arg_names[i];
                arg_names_.Add(Marshal.PtrToStringAnsi((IntPtr)pArgName));
            }
        }

        /// <summary>
        /// create a Symbol from the current operator
        /// </summary>
        /// <param name="name">the name of the operator</param>
        /// <returns>the operator Symbol</returns>
        public Symbol CreateSymbol(String name = null)
        {
            if (String.IsNullOrEmpty(name) == true) name = GetSymbolName(this._opName);

            if (input_keys.Count > 0)
            {
                Logging.CHECK_EQ(input_keys.Count, input_symbols.Count);
            }

            using (StringHolder hName = new StringHolder(name))
            using (StringListHolder hInputKeys = input_keys.GetHolder())
            using (StringListHolder hParamKeys = params_.Keys.GetHolder())
            using (StringListHolder hParamValues = params_.Values.GetHolder())
            using (IntPtrListHolder hInputSymbols = input_symbols.GetHolder())
            {
                byte* pName = hName.Handle;
                if (String.IsNullOrEmpty(name)) pName = null;

                SymbolHandle symbol_handle = new SymbolHandle();
                byte** pInputKeys = input_keys.Count > 0 ? hInputKeys.Pointer : null;

                Logging.CHECK_EQ(CAPI.MXSymbolCreateAtomicSymbol(handle_, (uint)params_.Count, hParamKeys.Pointer,
                                       hParamValues.Pointer, &symbol_handle), 0);
                Logging.CHECK_EQ(CAPI.MXSymbolCompose(symbol_handle, pName, (uint)input_symbols.Count, pInputKeys,
                                hInputSymbols.Handle), 0);
                return new Symbol(symbol_handle);
            }
        }

        public Symbol CreateSymbol(Symbol data, String name = null)
        {
            if(data != null) this.SetInput("data", data);
            return CreateSymbol(name);
        }

        public Symbol CreateSymbol(Symbol data, Symbol label, String name = null)
        {
            if (data != null) this.SetInput("data", data);
            if (label != null) this.SetInput("label", label);
            return CreateSymbol(name);
        }

        public void Invoke(List<NDArray> outputs)
        {
            if (input_keys.Count > 0)
            {
                Logging.CHECK_EQ(input_keys.Count, input_ndarrays.Count);
            }

            using (StringListHolder hInputKeys = input_keys.GetHolder())
            using (StringListHolder hParamKeys = params_.Keys.GetHolder())
            using (StringListHolder hParamValues = params_.Values.GetHolder())
            using (IntPtrListHolder hInputNDArrays = input_ndarrays.GetHolder())
            {
                int num_inputs = input_ndarrays.Count;
                int num_outputs = outputs.Count;
                NDArrayHandle[] output_handles = outputs.GetHandles();
                fixed (NDArrayHandle* pOutputs = output_handles)
                {
                    NDArrayHandle* pOutputsReceiver = null;
                    if (num_outputs > 0)
                    {
                        pOutputsReceiver = pOutputs;
                    }

                    CAPI.MXImperativeInvoke(handle_, num_inputs, hInputNDArrays.Handle,
                        &num_outputs, &pOutputsReceiver,
                        params_.Count, hParamKeys.Pointer, hParamValues.Pointer);

                    if (outputs.Count > 0)
                        return;

                    for (int i = 0; i < num_outputs; i++)
                    {
                        outputs.Add(new NDArray(pOutputsReceiver[i]));
                    }
                }
            }
        }

        public List<NDArray> Invoke()
        {
            List<NDArray> outputs = new List<NDArray>();
            Invoke(outputs);
            return outputs;
        }

        public void Invoke(NDArray output)
        {
            List<NDArray> outputs = new List<NDArray>();
            outputs.Add(output);
            Invoke(outputs);
        }
    }

    #region Executor

    public unsafe class Executor : IDisposable
    {
        ExecutorHandle _handle;
        Symbol _symbol;

        List<NDArray> outputs = new List<NDArray>();
        List<NDArray> arg_arrays = new List<NDArray>();
        List<NDArray> grad_arrays = new List<NDArray>();
        List<NDArray> aux_arrays = new List<NDArray>();

        public List<NDArray> Outputs
        {
            get { return outputs; }
        }

        public Executor(Symbol symbol, Context context,
           List<NDArray> arg_arrays,
           List<NDArray> grad_arrays,
           List<OpReqType> grad_reqs,
           List<NDArray> aux_arrays,
           Dictionary<String, Context> group_to_ctx,
           Executor shared_exec = null)
        {
            this.arg_arrays = arg_arrays;
            this.grad_arrays = grad_arrays;
            this.aux_arrays = aux_arrays;
            this._symbol = symbol;

            NDArrayHandle[] arg_handles = arg_arrays.GetHandles();
            NDArrayHandle[] grad_handles = grad_arrays.GetHandles();
            NDArrayHandle[] aux_handles = aux_arrays.GetHandles();
            mx_uint[] grad_reqs_uint = grad_reqs.GetValues();

            StringHolder[] map_keys = new StringHolder[group_to_ctx.Count];
            int[] dev_types = new int[map_keys.Length];
            int[] dev_ids = new int[map_keys.Length];

            int idx = 0;
            foreach (var item in group_to_ctx)
            {
                map_keys[idx] = new StringHolder(item.Key);
                Context cxt = item.Value;
                dev_types[idx] = (int)cxt.DeviceType;
                dev_ids[idx] = cxt.DeviceId;
                idx++;
            }

            ExecutorHandle sharedExecHandle = IntPtr.Zero;
            ExecutorHandle* pSharedExecHandle = null;
            if (shared_exec != null)
            {
                sharedExecHandle = shared_exec._handle;
                pSharedExecHandle = &sharedExecHandle;
            }

            ExecutorHandle handle = _handle;
            fixed (int* pDevTypes = dev_types)
            fixed (int* pDevIds = dev_ids)
            fixed (NDArrayHandle* pArgHandles = arg_handles)
            fixed (NDArrayHandle* pGradHandles = grad_handles)
            fixed (NDArrayHandle* pAuxHandles = aux_handles)
            fixed (mx_uint* pReqs = grad_reqs_uint)
            {
                IntPtr* pStrings = stackalloc IntPtr[map_keys.Length];
                for (int i = 0; i < map_keys.Length; i++)
                    pStrings[i] = (IntPtr)map_keys[i].Handle;

                Logging.CHECK_EQ(CAPI.MXExecutorBindEX(symbol.Handle, (int)context.DeviceType,
                           context.DeviceId, (uint)group_to_ctx.Count,
                           (Byte**)pStrings, pDevTypes, pDevIds,
                           (uint)arg_handles.Length, pArgHandles,
                           pGradHandles, pReqs,
                           (uint)aux_handles.Length, pAuxHandles,
                           (IntPtr)pSharedExecHandle, &handle),
                0);

                if (shared_exec != null)
                {
                    shared_exec._handle = sharedExecHandle;
                }
                _handle = handle;
            }

            foreach (var item in map_keys)
                item.Dispose();

            mx_uint out_size;
            NDArrayHandle* out_array;
            Logging.CHECK_EQ(CAPI.MXExecutorOutputs(_handle, &out_size, &out_array), 0);
            for (mx_uint i = 0; i < out_size; ++i)
            {
                outputs.Add(new NDArray(out_array[i]));
            }
        }

        public Executor(ExecutorHandle h) { _handle = h; }

        /// <summary>
        /// Perform a Forward operation of Operator
        /// After this operation, user can get the result by using function head.
        /// </summary>
        /// <param name="is_train"></param>
        public void Forward(bool is_train)
        {
            CAPI.MXExecutorForward(_handle, is_train ? 1 : 0);
            mx_uint out_size;
            NDArrayHandle* out_array;
            Logging.CHECK_EQ(CAPI.MXExecutorOutputs(_handle, &out_size, &out_array), 0);
            for (mx_uint i = 0; i < out_size; ++i)
            {
                outputs.Add(new NDArray(out_array[i]));
            }
        }

        /// <summary>
        /// Perform a Backward operation of the Operator.
        /// This must be called after Forward.
        /// After this operation, NDArrays specified by grad_in_args_store will be
        /// updated accordingly.
        /// User is allowed to pass in an empty Array if the head node is
        /// loss function and head gradeitn is not needed.
        /// </summary>
        /// <param name="head_grads">the gradient of head nodes to be backproped.</param>
        public void Backward(List<NDArray> head_grads = null)
        {
            int count = head_grads == null ? 0 : head_grads.Count;
            if (count > 0)
            {
                NDArrayHandle* pHandles = stackalloc NDArrayHandle[head_grads.Count];
                for (int i = 0; i < head_grads.Count; i++)
                {
                    pHandles[i] = head_grads[i].Handle;
                }
                CAPI.MXExecutorBackward(_handle, (uint)count, pHandles);
            }
            else
            {
                CAPI.MXExecutorBackward(_handle, 0, null);
            }
        }

        public void Reshape()
        {
            //TODO: Reshape
        }

        /// <summary>
        /// update the arguments with given learning rate and optimizer
        /// </summary>
        /// <returns></returns>
        public String DebugString()
        {
            Byte* output = null;
            CAPI.MXExecutorPrint(_handle, &output);
            return Marshal.PtrToStringAnsi((IntPtr)output);
        }

        public Dictionary<String, NDArray> arg_dict()
        {
            return GetDict(_symbol.ListArguments(), arg_arrays);
        }

        public Dictionary<String, NDArray> grad_dict()
        {
            return GetDict(_symbol.ListArguments(), grad_arrays);
        }

        public Dictionary<String, NDArray> aux_dict()
        {
            return GetDict(_symbol.ListAuxiliaryStates(), aux_arrays);
        }

        /// <summary>
        /// update the arguments with given learning rate and optimizer
        /// </summary>
        /// <param name="opt">the optimizer</param>
        /// <param name="lr">learning rate</param>
        /// <param name="wd">weight decay</param>
        /// <param name="arg_update_begin">begin index of the arguments to be updated, it starts after the input data by default</param>
        /// <param name="arg_update_end">end index of the arguments to be updated, it ends before the label data by default</param>
        public void UpdateAll(Optimizer opt, float lr, float wd, int arg_update_begin = 0,
                 int arg_update_end = -1)
        {
            arg_update_end = arg_update_end < 0 ? arg_arrays.Count - 1 : arg_update_end;
            for (int i = arg_update_begin; i < arg_update_end; ++i)
            {
                opt.Update(i, arg_arrays[i], grad_arrays[i], lr, wd);
            }
        }

        Dictionary<String, NDArray> GetDict(List<String> names,
                                         List<NDArray> arrays)
        {
            Dictionary<String, NDArray> ret = new Dictionary<string, NDArray>();
            HashSet<String> name_set = new HashSet<string>();
            foreach (var name in names)
            {
                if (name_set.Contains(name)) throw new Exception("Duplicate names detected:" + name_set);

                name_set.Add(name);
            }

            if (name_set.Count != arrays.Count)
                throw new Exception("names size not equal to arrays size");

            for (int i = 0; i < names.Count; i++)
            {
                ret[names[i]] = arrays[i];
            }
            return ret;
        }

        ~Executor()
        {
            Dispose();
        }

        public void Dispose()
        {
            if (_handle != IntPtr.Zero)
            {
                CAPI.MXExecutorFree(_handle);
                _handle = IntPtr.Zero;
            }
        }
    }

    #endregion

    #region initializers

    public class Initializer
    {
        public virtual void InitWeight(NDArray weight)
        {
        }
    }

    /// <summary>
    /// Initialize the weight with normal(0, sigma)
    /// </summary>
    public class NormalInitializer : Initializer
    {
        private float _mu = 0;
        private float _sigma;

        public NormalInitializer(float mu, float sigma)
        {
            this._mu = mu;
            this._sigma = sigma;
        }

        public NormalInitializer(float sigma)
        {
            this._sigma = sigma;
        }

        public override void InitWeight(NDArray weight)
        {
            NDArray.SampleGaussian(0, _sigma, weight);
        }
    }

    public class XavierInitializer : Initializer
    {
        public enum XavierType
        {
            Avg, In, Out
        }

        public enum RandomType
        {
            Uniform, Gaussian
        }

        private XavierType _type;
        private float _magnitude;
        private RandomType _rnType;

        public XavierInitializer(float magnitude, XavierType type = XavierType.In, RandomType randomType = RandomType.Gaussian)
        {
            this._magnitude = magnitude;
            this._type = type;
            this._rnType = randomType;
        }

        public override void InitWeight(NDArray weight)
        {
            List<uint> shape = weight.GetShape();
            float hw_scale = 1.0f;
            if(shape.Count > 2)
            {
                for (int i = 2; i < shape.Count; i++)
                    hw_scale *= shape[i];
            }
            float fan_in = shape[shape.Count > 1 ? 1 : 0] * hw_scale;
            float fan_out = shape[0] * hw_scale;
            float factor = 1.0f;
            if (_type == XavierType.Avg)
                factor = (fan_in + fan_out) / 2.0f;
            else if(_type == XavierType.In)
                    factor = fan_in;
            else
                factor = fan_out;
            float scale = (float)Math.Sqrt(_magnitude / factor);

            if (_rnType == RandomType.Gaussian)
                NDArray.SampleGaussian(0, scale, weight);
            else if (_rnType == RandomType.Uniform)
                NDArray.SampleUniform(-scale, scale, weight);
        }
    }

    #endregion

    #region Symbol

    /// <summary>
    /// Class to store NDArrayHandle
    /// </summary>
    public class SymBlob : IDisposable
    {
        private SymbolHandle _handle;

        public SymbolHandle Handle
        {
            get { return _handle; }
        }

        /// <summary>
        /// construct with a NDArrayHandle
        /// </summary>
        /// <param name="handle">handle NDArrayHandle to store</param>
        public SymBlob(SymbolHandle handle)
        {
            _handle = handle;
        }

        #region IDisposable Support
        private bool disposedValue = false; // To detect redundant calls

        ~SymBlob()
        {
            Dispose();
        }

        public void Dispose()
        {
            if (!disposedValue)
            {
                CAPI.MXSymbolFree(_handle);
                _handle = IntPtr.Zero;

                disposedValue = true;
            }
        }
        #endregion
    }

    public unsafe class Symbol
    {
        private static OpMap _opMap = new OpMap();

        private SymBlob blob_ptr_;

        public SymbolHandle Handle
        {
            get { return blob_ptr_.Handle; }
        }

        /// <summary>
        /// construct a Symbol with SymbolHandle
        /// </summary>
        /// <param name="handle">the given SymbolHandle</param>
        public Symbol(SymbolHandle handle)
        {
            blob_ptr_ = new SymBlob(handle);
        }

        /// <summary>
        /// construct a variable Symbol
        /// </summary>
        /// <param name="name">the name of the variable</param>
        public Symbol(String name)
        {
            SymbolHandle handle = new NDArrayHandle();
            StringHolder hName = new StringHolder(name);
            Logging.CHECK_EQ(CAPI.MXSymbolCreateVariable(hName.Handle, &(handle)), 0);
            blob_ptr_ = new SymBlob(handle);
        }

        #region operators overrided

        public static Symbol operator +(Symbol lhs, Symbol rhs)
        {
            return _Plus(lhs, rhs);
        }

        public static Symbol operator -(Symbol lhs, Symbol rhs)
        {
            return _Minus(lhs, rhs);
        }

        public static Symbol operator *(Symbol lhs, Symbol rhs)
        {
            return _Mul(lhs, rhs);
        }

        public static Symbol operator /(Symbol lhs, Symbol rhs)
        {
            return _Div(lhs, rhs);
        }

        public static Symbol operator +(Symbol lhs, mx_float scalar)
        {
            return _PlusScalar(lhs, scalar);
        }

        public static Symbol operator -(Symbol lhs, mx_float scalar)
        {
            return _MinusScalar(lhs, scalar);
        }

        public static Symbol operator *(Symbol lhs, mx_float scalar)
        {
            return _MulScalar(lhs, scalar);
        }

        public static Symbol operator /(Symbol lhs, mx_float scalar)
        {
            return _DivScalar(lhs, scalar);
        }

        public static Symbol operator +(mx_float scalar, Symbol rhs)
        {
            return rhs + scalar;
        }

        public static Symbol operator -(mx_float scalar, Symbol rhs)
        {
            return _RMinusScalar(scalar, rhs);
        }

        public static Symbol operator *(mx_float scalar, Symbol rhs)
        {
            return rhs * scalar;
        }

        public static Symbol operator /(mx_float scalar, Symbol rhs)
        {
            return _RDivScalar(scalar, rhs);
        }

        #endregion

        #region symbols

        public static Symbol _Plus(Symbol lhs, Symbol rhs)
        {
            return new Operator("_Plus").Set(lhs, rhs).CreateSymbol();
        }

        public static Symbol _Mul(Symbol lhs, Symbol rhs)
        {
            return new Operator("_Mul").Set(lhs, rhs).CreateSymbol();
        }

        public static Symbol _Minus(Symbol lhs, Symbol rhs)
        {
            return new Operator("_Minus").Set(lhs, rhs).CreateSymbol();
        }

        public static Symbol _Div(Symbol lhs, Symbol rhs)
        {
            return new Operator("_Div").Set(lhs, rhs).CreateSymbol();
        }

        public static Symbol _Power(Symbol lhs, Symbol rhs)
        {
            return new Operator("_Power").Set(lhs, rhs).CreateSymbol();
        }

        public static Symbol _Maximum(Symbol lhs, Symbol rhs)
        {
            return new Operator("_Maximum").Set(lhs, rhs).CreateSymbol();
        }

        public static Symbol _Minimum(Symbol lhs, Symbol rhs)
        {
            return new Operator("_Minimum").Set(lhs, rhs).CreateSymbol();
        }

        public static Symbol _PlusScalar(Symbol lhs, mx_float scalar)
        {
            return new Operator("_PlusScalar").Set(lhs).SetParam("scalar", scalar).CreateSymbol();
        }

        public static Symbol _MinusScalar(Symbol lhs, mx_float scalar)
        {
            return new Operator("_MinusScalar").Set(lhs).SetParam("scalar", scalar).CreateSymbol();
        }

        public static Symbol _RMinusScalar(mx_float scalar, Symbol rhs)
        {
            return new Operator("_RMinusScalar").Set(rhs)
                     .SetParam("scalar", scalar)
                     .CreateSymbol();
        }

        public static Symbol _MulScalar(Symbol lhs, mx_float scalar)
        {
            return new Operator("_MulScalar").Set(lhs)
                     .SetParam("scalar", scalar)
                     .CreateSymbol();
        }

        public static Symbol _DivScalar(Symbol lhs, mx_float scalar)
        {
            return new Operator("_DivScalar").Set(lhs)
                     .SetParam("scalar", scalar)
                     .CreateSymbol();
        }

        public static Symbol _RDivScalar(mx_float scalar, Symbol rhs)
        {
            return new Operator("_RDivScalar").Set(rhs)
                     .SetParam("scalar", scalar)
                     .CreateSymbol();
        }

        public static Symbol _PowerScalar(Symbol lhs, mx_float scalar)
        {
            return new Operator("_PowerScalar").Set(lhs)
                     .SetParam("scalar", scalar)
                     .CreateSymbol();
        }
        public static Symbol _RPowerScalar(mx_float scalar, Symbol rhs)
        {
            return new Operator("_RPowerScalar").Set(rhs)
                     .SetParam("scalar", scalar)
                     .CreateSymbol();
        }

        public static Symbol _MaximumScalar(Symbol lhs, mx_float scalar)
        {
            return new Operator("_MaximumScalar").Set(lhs)
                     .SetParam("scalar", scalar)
                     .CreateSymbol();
        }

        public static Symbol _MinimumScalar(Symbol lhs, mx_float scalar)
        {
            return new Operator("_MinimumScalar").Set(lhs)
                     .SetParam("scalar", scalar)
                     .CreateSymbol();
        }

        public static Symbol Crop(String symbol_name,
            int num_args,
            Symbol data,
            Symbol crop_like,
            Shape offset,
            Shape h_w,
            bool center_crop = false)
        {
            return new Operator("Crop")
              .SetParam("num_args", num_args)
              .SetParam("offset", offset)
              .SetParam("h_w", h_w)
              .SetParam("center_crop", center_crop)
              .SetInput("arg0", data)
              .SetInput("arg1", crop_like)
              .CreateSymbol(symbol_name);
        }

        public static Symbol SliceChannel(Symbol data,
                           int num_outputs,
                           int axis = 1,
                           bool squeeze_axis = false)
        {
            return new Operator("SliceChannel")
                .SetParam("num_outputs", num_outputs)
                .SetParam("axis", axis)
                .SetParam("squeeze_axis", squeeze_axis)
                .Set(data)
                .CreateSymbol();
        }

        #endregion

        /// <summary>
        /// Apply activation function to input.
        /// Softmax Activation is only available with CUDNN on GPUand will be
        /// computed at each location across channel if input is 4D.
        /// </summary>
        /// <param name="symbol_name">name of the resulting symbol.</param>
        /// <param name="data">Input data to activation function.</param>
        /// <param name="act_type">Activation function to be applied. </param>
        /// <returns>new symbol</returns>
        public static Symbol Activation(String symbol_name,
                         Symbol data,
                         String act_type)
        {
            System.Diagnostics.Debug.Assert(act_type == "relu" ||
                   act_type == "sigmoid" ||
                   act_type == "softrelu" ||
                   act_type == "tanh");
            return new Operator("Activation")
                     .SetParam("act_type", act_type)
                     .SetInput("data", data)
                     .CreateSymbol(symbol_name);
        }

        public static Operator Create(String opName)
        {
            return new Operator(opName);
        }

        public static Symbol Activation(String symbol_name,
                        Symbol data,
                        ActivationActType type)
        {
            return Activation(symbol_name, data, type.ToString());
        }

        public Symbol this[int index]
        {
            get
            {
                SymbolHandle hOut = new NDArrayHandle();
                CAPI.MXSymbolGetOutput(Handle, (uint)index, &hOut);
                return new Symbol(hOut);
            }
        }

        public Symbol this[String name]
        {
            get
            {
                var list = ListOutputs();
                for (int i = 0; i < list.Count; i++)
                {
                    if (list[i] == name) return this[i];
                }

                Logging.LOG_FATAL("Cannot find output that matches name " + name);
                return this[0];
            }
        }

        /// <summary>
        /// Create a symbol that groups symbols together
        /// </summary>
        /// <param name="symbols">List of symbols to be group</param>
        /// <returns></returns>
        public static Symbol Group(List<Symbol> symbols)
        {
            SymbolHandle pOut = new SymbolHandle();
            SymbolHandle[] handles = symbols.GetHandles();
            fixed (SymbolHandle* p = handles)
            {
                CAPI.MXSymbolCreateGroup((uint)symbols.Count, p, &pOut);
            }
            return new Symbol(pOut);
        }

        /// <summary>
        /// load Symbol from a JSON file
        /// </summary>
        /// <param name="fileName">the name of the file</param>
        /// <returns></returns>
        public static Symbol Load(String fileName)
        {
            SymbolHandle handle;
            Logging.CHECK_EQ(CAPI.MXSymbolCreateFromFile(fileName, &(handle)), 0);
            return new Symbol(handle);
        }

        public static Symbol LoadJSON(String jsonStr)
        {
            SymbolHandle handle;
            Logging.CHECK_EQ(CAPI.MXSymbolCreateFromJSON(jsonStr, &(handle)), 0);
            return new Symbol(handle);
        }

        public void Save(String fileName)
        {
            Logging.CHECK_EQ(CAPI.MXSymbolSaveToFile(Handle, fileName), 0);
        }

        public String ToJSON()
        {
            Byte* pOut;
            Logging.CHECK_EQ(CAPI.MXSymbolSaveToJSON(Handle, &pOut), 0);
            return Marshal.PtrToStringAnsi((IntPtr)pOut);
        }

        public override string ToString()
        {
            return ToJSON();
        }

        public Symbol GetInternals()
        {
            SymbolHandle handle;
            Logging.CHECK_EQ(CAPI.MXSymbolGetInternals(Handle, &handle), 0);
            return new Symbol(handle);
        }

        /// <summary>
        /// construct a variable Symbol
        /// </summary>
        /// <param name="name">the name of the variable</param>
        /// <returns></returns>
        public static Symbol Variable(String name)
        {
            return new Symbol(name);
        }

        public Symbol Clone()
        {
            SymbolHandle handle;
            Logging.CHECK_EQ(CAPI.MXSymbolCopy(Handle, &handle), 0);
            return new Symbol(handle);
        }

        /// <summary>
        /// List the arguments names.
        /// The position of the returned list also corresponds to calling position in operator()
        /// </summary>
        /// <returns>the arguments list of this symbol, they can be either named or unnamed (empty string).</returns>
        public List<String> ListArguments()
        {
            List<String> ret = new List<string>();
            mx_uint size;
            Byte** sarr;
            CAPI.MXSymbolListArguments(Handle, &size, &sarr);
            for (mx_uint i = 0; i < size; ++i)
            {
                ret.Add(Marshal.PtrToStringAnsi((IntPtr)(sarr[i])));
            }
            return ret;
        }

        /// <summary>
        /// get the descriptions of outputs for this symbol
        /// </summary>
        /// <returns></returns>
        public List<String> ListOutputs()
        {
            List<String> ret = new List<string>();
            mx_uint size;
            Byte** sarr;
            CAPI.MXSymbolListOutputs(Handle, &size, &sarr);
            for (mx_uint i = 0; i < size; ++i)
            {
                ret.Add(Marshal.PtrToStringAnsi((IntPtr)(sarr[i])));
            }
            return ret;
        }

        /// <summary>
        /// get the descriptions of auxiliary data for this symbol
        /// </summary>
        /// <returns></returns>
        public List<String> ListAuxiliaryStates()
        {
            List<String> ret = new List<string>();
            mx_uint size;
            Byte** sarr;
            CAPI.MXSymbolListAuxiliaryStates(Handle, &size, &sarr);
            for (mx_uint i = 0; i < size; ++i)
            {
                ret.Add(Marshal.PtrToStringAnsi((IntPtr)(sarr[i])));
            }
            return ret;
        }

        /// <summary>
        /// infer the shapes by providing shapes of known argument shapes.
        /// </summary>
        /// <param name="arg_shapes">map of argument name to shape of arguments with known shapes</param>
        /// <param name="in_shape">used to store infered shapes of input arguments.</param>
        /// <param name="aux_shape">used to store infered shapes of outputs.</param>
        /// <param name="out_shape">use to store the infered shapes of auxiliary states</param>
        public void InferShape(Dictionary<String, List<mx_uint>> arg_shapes,
            List<List<mx_uint>> in_shape,
            List<List<mx_uint>> aux_shape,
            List<List<mx_uint>> out_shape
            )
        {
            List<mx_uint> arg_ind_ptr = new List<mx_uint>();
            List<mx_uint> arg_shape_data = new List<mx_uint>();

            foreach (var item in arg_shapes.Values)
            {
                arg_ind_ptr.Add((uint)arg_shape_data.Count);
                foreach (var i in item)
                    arg_shape_data.Add(i);
            }
            arg_ind_ptr.Add((uint)arg_shape_data.Count);

            using (StringListHolder keys = arg_shapes.Keys.GetHolder())
            using (UInt32ListHolder hArgIndPtr = arg_ind_ptr.GetHolder())
            using (UInt32ListHolder hArgShapeData = arg_shape_data.GetHolder())
            {
                mx_uint in_shape_size;
                mx_uint* in_shape_ndim;
                mx_uint** in_shape_data;
                mx_uint out_shape_size;
                mx_uint* out_shape_ndim;
                mx_uint** out_shape_data;
                mx_uint aux_shape_size;
                mx_uint* aux_shape_ndim;
                mx_uint** aux_shape_data;
                int complete;

                Logging.CHECK_EQ(CAPI.MXSymbolInferShape(Handle, (uint)arg_shapes.Count, keys.Pointer,
                                            hArgIndPtr.Handle, hArgShapeData.Handle,
                                            &in_shape_size, &in_shape_ndim, &in_shape_data,
                                            &out_shape_size, &out_shape_ndim, &out_shape_data,
                                            &aux_shape_size, &aux_shape_ndim, &aux_shape_data,
                                            &complete),
                         0);

                if (complete != 0)
                {
                    for (mx_uint i = 0; i < in_shape_size; ++i)
                    {
                        in_shape.Add(new List<mx_uint>());
                        for (mx_uint j = 0; j < in_shape_ndim[i]; ++j)
                        {
                            in_shape[(int)i].Add(in_shape_data[i][j]);
                        }
                    }
                    for (mx_uint i = 0; i < aux_shape_size; ++i)
                    {
                        aux_shape.Add(new List<mx_uint>());
                        for (mx_uint j = 0; j < aux_shape_ndim[i]; ++j)
                        {
                            aux_shape[(int)i].Add(aux_shape_data[i][j]);
                        }
                    }
                    for (mx_uint i = 0; i < out_shape_size; ++i)
                    {
                        out_shape.Add(new List<mx_uint>());
                        for (mx_uint j = 0; j < out_shape_ndim[i]; ++j)
                        {
                            out_shape[(int)i].Add(out_shape_data[i][j]);
                        }
                    }
                }
            }
        }

        /// <summary>
        /// infer and construct all the arrays to bind to executor by providing some known arrays.
        /// </summary>
        /// <param name="context">the context of all the infered arrays</param>
        /// <param name="arg_arrays">infered input arguments arrays.</param>
        /// <param name="grad_arrays">infered arrays to store the gradient output of the input arguments</param>
        /// <param name="grad_reqs"></param>
        /// <param name="aux_arrays">infered arrays that is used as internal state in op.</param>
        /// <param name="args_map">map of some given arguments arrays.</param>
        /// <param name="arg_grad_store">map of some gradient given store arrays.</param>
        /// <param name="grad_req_type">map of some given type of gradient saving. Can only be in {kNullOp, kAddTo, kWriteTo}.</param>
        /// <param name="aux_map">NDArray that stores the internal state in op</param>
        public void InferExecutorArrays(Context context,
            List<NDArray> arg_arrays,
            List<NDArray> grad_arrays,
            List<OpReqType> grad_reqs,
            List<NDArray> aux_arrays,
            Dictionary<String, NDArray> args_map,
            Dictionary<String, NDArray> arg_grad_store,
            Dictionary<String, OpReqType> grad_req_type,
            Dictionary<String, NDArray> aux_map,
            Initializer initializer)
        {
            List<String> arg_name_list = ListArguments();
            List<List<mx_uint>> in_shapes = new List<List<mx_uint>>();
            List<List<mx_uint>> aux_shapes = new List<List<mx_uint>>();
            List<List<mx_uint>> out_shapes = new List<List<mx_uint>>();
            Dictionary<String, List<mx_uint>> arg_shapes = new Dictionary<string, List<mx_uint>>();

            foreach (String arg_name in arg_name_list)
            {
                if (args_map.ContainsKey(arg_name))
                {
                    arg_shapes[arg_name] = args_map[arg_name].GetShape();
                }
            }

            InferShape(arg_shapes, in_shapes, aux_shapes, out_shapes);

            for (int i = 0; i < in_shapes.Count; ++i)
            {
                var shape = in_shapes[i];
                var arg_name = arg_name_list[i];
                if (args_map.ContainsKey(arg_name))
                    arg_arrays.Add(args_map[arg_name]);
                else
                {
                    NDArray nd = new NDArray(shape, context, false);
                    arg_arrays.Add(nd);
                    initializer.InitWeight(nd);
                }

                if (arg_grad_store.ContainsKey(arg_name))
                    grad_arrays.Add(arg_grad_store[arg_name]);
                else
                    grad_arrays.Add(new NDArray(shape, context, false));

                if (grad_req_type.ContainsKey(arg_name))
                    grad_reqs.Add(grad_req_type[arg_name]);
                else if (arg_name.EndsWith("data"))
                    grad_reqs.Add(OpReqType.kNullOp);
                else
                    grad_reqs.Add(OpReqType.kWriteTo);
            }

            var aux_name_list = ListAuxiliaryStates();
            for (int i = 0; i < aux_shapes.Count; ++i)
            {
                var shape = aux_shapes[i];
                var aux_name = aux_name_list[i];
                if (aux_map.ContainsKey(aux_name))
                {
                    aux_arrays.Add(aux_map[aux_name]);
                }
                else
                {
                    NDArray nd = new NDArray(shape, context, false);
                    initializer.InitWeight(nd);
                    aux_arrays.Add(nd);
                }
            }
        }

        /// <summary>
        /// infer and construct all the input arguments arrays to bind to 
        /// executor by providing some known arguments arrays.
        /// </summary>
        /// <param name="context">the context of all the infered arrays.</param>
        /// <param name="args_map">map of all the infered input arguments arrays.</param>
        /// <param name="known_args">map of some given arguments arrays.</param>
        public void InferArgsMap(Context context,
            Dictionary<String, NDArray> args_map,
            Dictionary<String, NDArray> known_args, Initializer initializer)
        {
            var arg_name_list = ListArguments();
            List<List<mx_uint>> in_shapes = new List<List<mx_uint>>();
            List<List<mx_uint>> aux_shapes = new List<List<mx_uint>>();
            List<List<mx_uint>> out_shapes = new List<List<mx_uint>>();
            Dictionary<String, List<mx_uint>> arg_shapes = new Dictionary<string, List<mx_uint>>();

            foreach (String arg_name in arg_name_list)
            {
                if (args_map.ContainsKey(arg_name))
                {
                    arg_shapes[arg_name] = args_map[arg_name].GetShape();
                }
            }

            InferShape(arg_shapes, in_shapes, aux_shapes, out_shapes);

            for (int i = 0; i < in_shapes.Count; ++i)
            {
                var shape = in_shapes[i];
                var arg_name = arg_name_list[i];
                if (known_args != null && known_args.ContainsKey(arg_name))
                {
                    args_map[arg_name] = known_args[arg_name];
                }
                else
                {
                    NDArray nd = new NDArray(shape, context, false);
                    initializer.InitWeight(nd);
                    args_map[arg_name] = nd;
                }
            }
        }

        /// <summary>
        /// Create an executor by bind symbol with context and arguments.
        /// If user do not want to compute the gradients of i-th argument,
        /// grad_req_type[i] can be kNullOp.
        /// The input arrays in the given maps should have the same name with the input
        /// symbol.
        /// Only need some of the necessary arrays, and the other arrays can be infered
        /// automatically.
        /// </summary>
        /// <param name="context">the context of binding.</param>
        /// <param name="args_map">the NDArray that stores the input arguments to the symbol.</param>
        /// <param name="arg_grad_store">NDArray that is used to store the gradient output of the input arguments.</param>
        /// <param name="grad_req_type">requirment type of gradient saving. Can only be in {kNullOp, kAddTo, kWriteTo}.</param>
        /// <param name="aux_map">NDArray that stores the internal state in op </param>
        /// <returns>a new executor, which need to be free manually.</returns>
        public Executor SimpleBind(Context context,
                       Dictionary<String, NDArray> args_map,
                       Dictionary<String, NDArray> arg_grad_store,
                       Dictionary<String, OpReqType> grad_req_type,
                       Dictionary<String, NDArray> aux_map,
                       Initializer initializer)
        {
            List<NDArray> arg_arrays = new List<NDArray>();
            List<NDArray> grad_arrays = new List<NDArray>();
            List<OpReqType> grad_reqs = new List<OpReqType>();
            List<NDArray> aux_arrays = new List<NDArray>();

            InferExecutorArrays(context, arg_arrays, grad_arrays, grad_reqs,
                      aux_arrays, args_map, arg_grad_store, grad_req_type,
                      aux_map, initializer);

            return new Executor(this, context, arg_arrays, grad_arrays, grad_reqs,
                                aux_arrays, new Dictionary<string, Context>());
        }

        public Executor SimpleBind(Context context,
                       Dictionary<String, NDArray> args_map,
                       Initializer initializer)
        {
            return SimpleBind(context, args_map, new Dictionary<string, NDArray>(), new Dictionary<string, OpReqType>(), new Dictionary<string, NDArray>(), initializer);
        }

        /// <summary>
        /// Create an executor by bind symbol with context and arguments.
        /// If user do not want to compute the gradients of i-th argument,
        /// grad_req_type[i] can be kNullOp.
        /// </summary>
        /// <param name="context">the context of binding.</param>
        /// <param name="arg_arrays">the NDArray that stores the input arguments to the symbol.</param>
        /// <param name="grad_arrays">NDArray that is used to store the gradient output of the input arguments.</param>
        /// <param name="grad_reqs">requirment type of gradient saving. Can only be in {kNullOp, kAddTo, kWriteTo}.</param>
        /// <param name="aux_arrays">NDArray that is used as internal state in op </param>
        /// <param name="group_to_ctx">dict of string to mx.Context</param>
        /// <param name="shared_exec">
        /// Executor to share memory with. This is intended for
        /// runtime reshaping, variable length sequencesn etc.  The returned executor
        /// shares state with shared_exec, and should not be used in parallel with it.
        /// </param>
        /// <returns>a new executor, which need to be free manually.</returns>
        public Executor Bind(Context context,
            List<NDArray> arg_arrays,
            List<NDArray> grad_arrays,
            List<OpReqType> grad_reqs,
            List<NDArray> aux_arrays,
            Dictionary<String, Context> group_to_ctx,
            Executor shared_exec = null)
        {
            return new Executor(this, context, arg_arrays, grad_arrays, grad_reqs,
                      aux_arrays, group_to_ctx, shared_exec);
        }
    }

    #endregion

    public class Optimizer : IDisposable
    {
        protected static OpMap op_map_ = new OpMap();

        protected Dictionary<String, String> params_ = new Dictionary<string, string>();

        public Object this[String key]
        {
            get { return params_[key]; }
            set { params_[key] = value.ToString(); }
        }

        public Optimizer SetParam(String key, Object value)
        {
            params_[key] = value.ToString();
            return this;
        }

        protected List<String> GetParamKeys_()
        {
            List<String> list = new List<string>();
            list.AddRange(params_.Keys);
            return list;
        }

        protected List<String> GetParamValues_()
        {
            List<String> list = new List<string>();
            list.AddRange(params_.Values);
            return list;
        }

        public virtual String GetType()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Update a weight with gradient.
        /// </summary>
        /// <param name="index">the unique index for the weight.</param>
        /// <param name="weight">the weight to update.</param>
        /// <param name="grad">gradient for the weight.</param>
        /// <param name="lr">learning rate.</param>
        /// <param name="wd">weight decay.</param>
        public void Update(int index, NDArray weight, NDArray grad, mx_float lr,
              mx_float wd)
        {
            params_["lr"] = lr.ToString();
            params_["wd"] = wd.ToString();
            Update(index, weight, grad);
        }

        public virtual void Update(int index, NDArray weight, NDArray grad)
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Serialize the optimizer parameters to a string.
        /// </summary>
        /// <returns>serialization</returns>
        public String Serialize()
        {
            params_["opt_type"] = GetType();
            StringBuilder sb = new StringBuilder();
            foreach(var pair in params_)
            {
                if (sb.Length > 0) sb.Append('\n');
                sb.Append(pair.Key).Append('=').Append(pair.Value);
            }
            return sb.ToString();
        }

        public virtual void Dispose()
        {
        }

        ~Optimizer()
        {
            Dispose();
        }
    }

    public unsafe class SGDOptimizer : Optimizer
    {
        AtomicSymbolCreator hSgdUpdate;
        AtomicSymbolCreator hSgdMomUpdate;
        Dictionary<int, NDArray> states_ = new Dictionary<int, NDArray>();

        public override string GetType()
        {
            return "sgd";
        }

        public SGDOptimizer():base()
        {
            hSgdUpdate = op_map_.GetSymbolCreator("sgd_update");
            hSgdMomUpdate = op_map_.GetSymbolCreator("sgd_mom_update");
        }

        public override void Dispose()
        {
            if (states_ == null) return;
            foreach(var item in states_.Values)
            {
                item.Dispose();
            }
            states_ = null;
        }

        public override void Update(int index, NDArray weight, NDArray grad)
        {
            if (states_.ContainsKey(index) == false)
            {
                CreateState(index, weight);
            }

            var keys = GetParamKeys_();
            var values = GetParamValues_();
            Logging.CHECK_EQ(keys.Count, values.Count);

            NDArrayHandle[] inputs = new NDArrayHandle[3];
            inputs[0] = weight.Handle;
            inputs[1] = grad.Handle;

            int num_outputs = 1;
            NDArrayHandle output = weight.Handle;
            NDArrayHandle* outputs = &output;

            using (StringListHolder hKeys = keys.GetHolder())
            using (StringListHolder hValues = values.GetHolder())
            {
                fixed(IntPtr* pInputs = inputs)
                {
                    if (states_[index] == null)
                    {
                        Logging.CHECK_EQ(CAPI.MXImperativeInvoke(hSgdUpdate, 2, pInputs,
                            &num_outputs, &outputs,
                            keys.Count, hKeys.Pointer, hValues.Pointer),0);
                    }
                    else
                    {
                        inputs[2] = states_[index].Handle;
                        Logging.CHECK_EQ(CAPI.MXImperativeInvoke(hSgdMomUpdate, 3, pInputs,
                            &num_outputs, &outputs,
                              keys.Count, hKeys.Pointer, hValues.Pointer),0);
                    }
                }
            }
        }

        private void CreateState(int index, NDArray weight)
        {
            if (params_.ContainsKey("momentum") == false)
            {
                states_[index] = null;
            }
            else
            {
                states_[index] = new NDArray(weight.GetShape(), weight.GetContext());
                states_[index].SetValue(0);
            }
        }
    }

    public class OptimizerRegistry
    {
        public static Optimizer Find(String name)
        {
            switch(name)
            {
                case "sgd":
                case "ccsgd":
                default:
                    return new SGDOptimizer();
            }
        }
    }
}
