using System;
using System.Collections.Generic;
using System.Text;
using System.Runtime.InteropServices;

namespace MXNetSharp
{
    /* ����CAPI.cs */
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

    public static class ClassHelper
    {
        public static NDArrayHandle[] GetHandles(this List<NDArray> list)
        {
            NDArrayHandle[] handles = new NDArrayHandle[list.Count];
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

        public static bool operator ==(Shape a, Shape b)
        {
            if (a._ndim != b._ndim) return false;
            else
            {
                for (int i = 0; i < a._ndim; i++)
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
                if(_blob!=null)
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
        Dictionary<String, AtomicSymbolCreator> symbol_creators_ = new Dictionary<string, NDArrayHandle>();
        Dictionary<String, OpHandle> op_handles_ = new Dictionary<string, NDArrayHandle>();

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
    }

    public class Operator
    {
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
            foreach(var item in group_to_ctx)
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
            fixed(int* pDevTypes = dev_types)
            fixed(int* pDevIds = dev_ids)
            fixed(NDArrayHandle* pArgHandles = arg_handles)
            fixed (NDArrayHandle* pGradHandles = grad_handles)
            fixed (NDArrayHandle* pAuxHandles = aux_handles)
            fixed(mx_uint* pReqs = grad_reqs_uint)
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

                if(shared_exec !=null)
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
        public void Backward(List<NDArray> head_grads)
        {
            int count = head_grads.Count;
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

        public Dictionary<String,NDArray> arg_dict()
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
        public void UpdateAll(Optimizer opt, float lr, float wd, int arg_update_begin = 1,
                 int arg_update_end = -1)
        {
            arg_update_end = arg_update_end < 0 ? arg_arrays.Count - 1 : arg_update_end;
            for (int i = arg_update_begin; i < arg_update_end; ++i)
            {
                opt.Update(i, arg_arrays[i], grad_arrays[i], lr, wd);
            }
        }

        Dictionary<String, NDArray> GetDict(List<String> names,
                                         List<NDArray> arrays) {
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
            if(_handle != IntPtr.Zero)
            {
                CAPI.MXExecutorFree(_handle);
                _handle = IntPtr.Zero;
            }
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

    public class Symbol
    {
        public SymbolHandle Handle
        {
            get { throw new NotImplementedException(); }
        }

        /// <summary>
        /// List the arguments names.
        /// The position of the returned list also corresponds to calling position in operator()
        /// </summary>
        /// <returns>the arguments list of this symbol, they can be either named or unnamed (empty string).</returns>
        public List<String> ListArguments()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// get the descriptions of outputs for this symbol
        /// </summary>
        /// <returns></returns>
        public List<String> ListOutputs()
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// get the descriptions of auxiliary data for this symbol
        /// </summary>
        /// <returns></returns>
        public List<String> ListAuxiliaryStates()
        {
            throw new NotImplementedException();
        }
    }

    #endregion

    public class Optimizer
    {
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
            throw new NotImplementedException();
        }

        public virtual void Update(int index, NDArray weight, NDArray grad)
        {
            throw new NotImplementedException();
        }
    }
}
