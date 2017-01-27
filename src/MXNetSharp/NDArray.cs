using System;
using System.Collections.Generic;
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

    public enum DeviceType
    {
        kCPU = 1,
        kGPU = 2,
        kCPUPinned = 3
    }

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
        public static Context Gpu(int device_id = 0) {
            return new Context(DeviceType.kGPU, device_id);
        }

        /// <summary>
        /// Return a CPU context
        /// </summary>
        /// <param name="device_id">id of the device. this is not needed by CPU</param>
        /// <returns>the corresponding CPU context</returns>
        public static Context Cpu(int device_id = 0) {
            return new Context(DeviceType.kCPU, device_id);
        }
    }

    /// <summary>
    /// Class to store NDArrayHandle
    /// </summary>
    public class NDBlob : IDisposable
    {
        private NDArrayHandle _handle;

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

    public unsafe class NDArray
    {
        private NDBlob _blob;

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
            :this(shape.Data,context,delay_alloc)
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
            fixed(mx_float* pDataArr = dataArr)
            {
                NDArrayHandle handle;
                Logging.CHECK_EQ(CAPI.MXNDArrayCreate(pArr, (uint)shape.NDim, (int)context.DeviceType,
                                         context.DeviceId, 0, &handle),
                         0);
                CAPI.MXNDArraySyncCopyFromCPU(handle, pDataArr, shape.Size);
                _blob = new NDBlob(handle);
            }
        }

        public NDArray(mx_float* data, Shape shape, Context context) {

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


    }
}
