using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

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
    using nn_uint = UInt32;
    using OpHandle = IntPtr;
    using GraphHandle = IntPtr;

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
}
