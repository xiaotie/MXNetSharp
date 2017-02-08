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
    using index_t = UInt32;
    using nn_uint = UInt32;
    using OpHandle = IntPtr;
    using GraphHandle = IntPtr;
    using size_t = UInt64;

    public unsafe class CAPI
    {
        public const String MXNET_DLL = "libmxnet";

        public delegate IntPtr ExecutorMonitorCallback(Byte* pChars, NDArrayHandle handle, void * p);

        [DllImport(MXNET_DLL)]
        public static extern String MXGetLastError();

        #region NDArray

        /* Part 1: NDArray creation and deletion */

        /// <summary>
        /// Create a NDArray handle that is not initialized can be used to pass in as mutate variables to hold the result of NDArray
        /// </summary>
        /// <param name="pOut">the returning handle</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXNDArrayCreateNone(NDArrayHandle* pOut);

        /// <summary>
        /// create a NDArray with specified shape
        /// </summary>
        /// <param name="shape">the pointer to the shape</param>
        /// <param name="ndim">the dimension of the shape</param>
        /// <param name="dev_type">device type, specify device we want to take</param>
        /// <param name="dev_id">the device id of the specific device</param>
        /// <param name="delay_alloc">whether to delay allocation until the narray is first mutated</param>
        /// <param name="pOut">the returning handle</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXNDArrayCreate(mx_uint* shape,
                              mx_uint ndim,
                              int dev_type,
                              int dev_id,
                              int delay_alloc,
                              NDArrayHandle* pOut);

        /// <summary>
        /// create a NDArray with specified shape and data type
        /// </summary>
        /// <param name="shape">the pointer to the shape</param>
        /// <param name="ndim">the dimension of the shape</param>
        /// <param name="dev_type">device type, specify device we want to take</param>
        /// <param name="dev_id">the device id of the specific device</param>
        /// <param name="delay_alloc">whether to delay allocation until the narray is first mutated</param>
        /// <param name="dtype">data type of created array</param>
        /// <param name="pOut">the returning handle</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXNDArrayCreateEx(mx_uint* shape,
                              mx_uint ndim,
                              int dev_type,
                              int dev_id,
                              int delay_alloc,
                              int dtype,
                              NDArrayHandle* pOut);


        /// <summary>
        /// create a NDArray handle that is loaded from raw bytes.
        /// </summary>
        /// <param name="buf">the head of the raw bytes</param>
        /// <param name="size">size of the raw bytes</param>
        /// <param name="pOut">the returning handle</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXNDArrayLoadFromRawBytes(void* buf,
                                        size_t size,
                                        NDArrayHandle* pOut);

        /// <summary>
        /// save the NDArray into raw bytes.
        /// </summary>
        /// <param name="handle">the NDArray handle</param>
        /// <param name="out_size"> size of the raw bytes</param>
        /// <param name="out_buf">the head of returning memory bytes</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXNDArraySaveRawBytes(NDArrayHandle handle,
                                    size_t* out_size,
                                    Byte** out_buf);

        /// <summary>
        /// Save list of narray into the file.
        /// </summary>
        /// <param name="fname">name of the file.</param>
        /// <param name="num_args">number of arguments to save.</param>
        /// <param name="args">the array of NDArrayHandles to be saved.</param>
        /// <param name="keys">the name of the NDArray, optional, can be NULL</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXNDArraySave(String fname,
                           mx_uint num_args,
                            NDArrayHandle* args,
                            Byte** keys = null);

        /// <summary>
        /// Load list of narray from the file.
        /// </summary>
        /// <param name="fname">name of the file.</param>
        /// <param name="out_size">number of narray loaded.</param>
        /// <param name="out_arr">head of the returning narray handles.</param>
        /// <param name="out_name_size">size of output name arrray.</param>
        /// <param name="out_names">the names of returning NDArrays, can be NULL</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXNDArrayLoad(String fname,
                            mx_uint *out_size,
                            NDArrayHandle** out_arr,
                            mx_uint *out_name_size,
                            byte*** out_names);

        /// <summary>
        /// Perform a synchronize copy from a continugous CPU memory region.
        /// This function will call WaitToWrite before the copy is performed.
        /// This is useful to copy data from existing memory region that are
        /// not wrapped by NDArray(thus dependency not being tracked).
        /// </summary>
        /// <param name="handle">the NDArray handle</param>
        /// <param name="data">the data source to copy from</param>
        /// <param name="size">the memory size we want to copy from.</param>
        /// <returns></returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXNDArraySyncCopyFromCPU(NDArrayHandle handle,
                                       void* data,
                                       size_t size);

        /// <summary>
        /// Perform a synchronize copyto a continugous CPU memory region.
        /// This function will call WaitToRead before the copy is performed.
        /// This is useful to copy data from existing memory region that are
        /// not wrapped by NDArray(thus dependency not being tracked).
        /// </summary>
        /// <param name="handle">the NDArray handle</param>
        /// <param name="data">the data source to copy into.</param>
        /// <param name="size">the memory size we want to copy into.</param>
        /// <returns></returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXNDArraySyncCopyToCPU(NDArrayHandle handle,
                                     void* data,
                                     size_t size);

        /// <summary>
        /// Wait until all the pending writes with respect NDArray are finished.
        /// Always call this before read data out synchronizely.
        /// </summary>
        /// <param name="handle">the NDArray handle</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXNDArrayWaitToRead(NDArrayHandle handle);

        /// <summary>
        /// Wait until all the pending read/write with respect NDArray are finished.
        /// Always call this before write data into NDArray synchronizely.
        /// </summary>
        /// <param name="handle">the NDArray handle</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXNDArrayWaitToWrite(NDArrayHandle handle);

        /// <summary>
        /// wait until all delayed operations in the system is completed
        /// </summary>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXNDArrayWaitAll();

        /// <summary>
        /// free the narray handle
        /// </summary>
        /// <param name="handle">the handle to be freed</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXNDArrayFree(NDArrayHandle handle);

        /// <summary>
        /// Slice the NDArray along axis 0.
        /// </summary>
        /// <param name="handle">the handle to the NDArray</param>
        /// <param name="slice_begin">The beginning index of slice</param>
        /// <param name="slice_end">The ending index of slice</param>
        /// <param name="pOut">The NDArrayHandle of sliced NDArray</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXNDArraySlice(NDArrayHandle handle,
                             mx_uint slice_begin,
                             mx_uint slice_end,
                             NDArrayHandle* pOut);

        /// <summary>
        /// Index the NDArray along axis 0.
        /// </summary>
        /// <param name="handle">the handle to the NDArray</param>
        /// <param name="idx">the index</param>
        /// <param name="pOut">The NDArrayHandle of output NDArray</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXNDArrayAt(NDArrayHandle handle,
                          mx_uint idx,
                          NDArrayHandle* pOut);

        /// <summary>
        /// Reshape the NDArray.
        /// </summary>
        /// <param name="handle">the handle to the narray</param>
        /// <param name="ndim">number of dimensions of new shape</param>
        /// <param name="dims">new shape</param>
        /// <param name="pOut">the NDArrayHandle of reshaped NDArray</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXNDArrayReshape(NDArrayHandle handle,
                               int ndim,
                               int* dims,
                               NDArrayHandle* pOut);

        /// <summary>
        /// get the shape of the array
        /// </summary>
        /// <param name="handle">the handle to the narray</param>
        /// <param name="out_dim">the output dimension</param>
        /// <param name="out_pdata">pointer holder to get data pointer of the shape</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXNDArrayGetShape(NDArrayHandle handle,
                                mx_uint* out_dim,
                                mx_uint** out_pdata);

        /// <summary>
        /// get the content of the data in NDArray
        /// </summary>
        /// <param name="handle">the handle to the narray</param>
        /// <param name="out_pdata">pointer holder to get pointer of data</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXNDArrayGetData(NDArrayHandle handle,
                               mx_float** out_pdata);

        /// <summary>
        /// get the type of the data in NDArray
        /// </summary>
        /// <param name="handle">the handle to the narray</param>
        /// <param name="out_dtype">pointer holder to get type of data</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXNDArrayGetDType(NDArrayHandle handle,
                               int* out_dtype);

        /// <summary>
        /// get the context of the NDArray
        /// </summary>
        /// <param name="handle">the handle to the narray</param>
        /// <param name="out_dev_type">the output device type</param>
        /// <param name="out_dev_id">the output device id</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXNDArrayGetContext(NDArrayHandle handle,
                                  int* out_dev_type,
                                  int* out_dev_id);

        /// <summary>
        /// invoke a nnvm op and imperative function
        /// </summary>
        /// <param name="creator">the op</param>
        /// <param name="num_inputs">number of input NDArrays</param>
        /// <param name="inputs">input NDArrays</param>
        /// <param name="num_outputs">number of output NDArrays</param>
        /// <param name="outputs">output NDArrays</param>
        /// <param name="num_params">number of keyword parameters</param>
        /// <param name="param_keys">keys for keyword parameters</param>
        /// <param name="param_vals">values for keyword parameters</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXImperativeInvoke(AtomicSymbolCreator creator,
                                 int num_inputs,
                                 NDArrayHandle* inputs,
                                 int* num_outputs,
                                 NDArrayHandle** outputs,
                                 int num_params,
                                 byte** param_keys,
                                 byte** param_vals);

        #endregion

        #region symbolic configuration generation

        /// <summary>
        /// list all the available operator names, include entries
        /// </summary>
        /// <param name="out_size">the size of returned array</param>
        /// <param name="out_array">the output operator name array</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXListAllOpNames(mx_uint* out_size,
                               byte*** out_array);

        /// <summary>
        /// list all the available AtomicSymbolEntry
        /// </summary>
        /// <param name="out_size">the size of returned array</param>
        /// <param name="out_array">the output AtomicSymbolCreator array</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXSymbolListAtomicSymbolCreators(mx_uint* out_size,
                                               AtomicSymbolCreator** out_array);

        /// <summary>
        /// Get the name of an atomic symbol.
        /// </summary>
        /// <param name="creator">the AtomicSymbolCreator.</param>
        /// <param name="name">The returned name of the creator.</param>
        /// <returns></returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXSymbolGetAtomicSymbolName(AtomicSymbolCreator creator,
                                          byte** name);

        /// <summary>
        /// Get the detailed information about atomic symbol.
        /// </summary>
        /// <param name="creator">creator the AtomicSymbolCreator.</param>
        /// <param name="name">The returned name of the creator.</param>
        /// <param name="description">The returned description of the symbol.</param>
        /// <param name="num_args">Number of arguments.</param>
        /// <param name="arg_names"> Name of the arguments.</param>
        /// <param name="arg_type_infos">Type informations about the arguments</param>
        /// <param name="arg_descriptions">Description information about the arguments</param>
        /// <param name="key_var_num_args">The keyword argument for specifying variable number of arguments.
        ///             When this parameter has non-zero length, the function allows variable number
        ///             of positional arguments, and will need the caller to pass it in in
        ///             MXSymbolCreateAtomicSymbol,
        ///             With key = key_var_num_args, and value = number of positional arguments.
        /// </param>
        /// <param name="return_type"> Return type of the function, can be Symbol or Symbol[]</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXSymbolGetAtomicSymbolInfo(AtomicSymbolCreator creator,
                                          byte** name,
                                          byte** description,
                                          mx_uint* num_args,
                                          byte*** arg_names,
                                          byte*** arg_type_infos,
                                          byte*** arg_descriptions,
                                          byte** key_var_num_args,
                                          byte** return_type = null);


        /// <summary>
        /// Create an AtomicSymbol.
        /// </summary>
        /// <param name="creator">the AtomicSymbolCreator</param>
        /// <param name="num_param">the number of parameters</param>
        /// <param name="keys">the keys to the params</param>
        /// <param name="vals">the vals of the params</param>
        /// <param name="pOut">pointer to the created symbol handle</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXSymbolCreateAtomicSymbol(AtomicSymbolCreator creator,
                                         mx_uint num_param,
                                         byte** keys,
                                         byte** vals,
                                         SymbolHandle* pOut);

        /// <summary>
        /// Create a Variable Symbol.
        /// </summary>
        /// <param name="name">name of the variable</param>
        /// <param name="pOut">pointer to the created symbol handle</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXSymbolCreateVariable(byte* name, SymbolHandle* pOut);

        /// <summary>
        /// Create a Symbol by grouping list of symbols together
        /// </summary>
        /// <param name="num_symbols">num_symbols number of symbols to be grouped</param>
        /// <param name="symbols">symbols array of symbol handles</param>
        /// <param name="pOut">pointer to the created symbol handle</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXSymbolCreateGroup(mx_uint num_symbols,
                                  SymbolHandle* symbols,
                                  SymbolHandle* pOut);

        /// <summary>
        /// Load a symbol from a json file.
        /// </summary>
        /// <param name="fname">the file name.</param>
        /// <param name="pOut">the output symbol.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXSymbolCreateFromFile(String fname, SymbolHandle* pOut);

        /// <summary>
        /// Load a symbol from a json string.
        /// </summary>
        /// <param name="json">the json string</param>
        /// <param name="pOut">the output symbol</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXSymbolCreateFromJSON(String json, SymbolHandle* pOut);

        /// <summary>
        /// Save a symbol into a json file.
        /// </summary>
        /// <param name="symbol">the input symbol.</param>
        /// <param name="fname">the file name.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXSymbolSaveToFile(SymbolHandle symbol, String fname);

        /// <summary>
        /// Save a symbol into a json string
        /// </summary>
        /// <param name="symbol">the input symbol.</param>
        /// <param name="out_json">output json string.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXSymbolSaveToJSON(SymbolHandle symbol, byte** out_json);

        /// <summary>
        /// Free the symbol handle.
        /// </summary>
        /// <param name="symbol">the symbol</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXSymbolFree(SymbolHandle symbol);

        /// <summary>
        /// Copy the symbol to another handle
        /// </summary>
        /// <param name="symbol">the source symbol</param>
        /// <param name="pOut"> used to hold the result of copy</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXSymbolCopy(SymbolHandle symbol, SymbolHandle* pOut);

        /// <summary>
        /// Print the content of symbol, used for debug.
        /// </summary>
        /// <param name="symbol">the symbol</param>
        /// <param name="out_str">pointer to hold the output string of the printing.</param>
        /// <returns> 0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXSymbolPrint(SymbolHandle symbol, byte** out_str);

        /// <summary>
        ///  Get string name from symbol
        /// </summary>
        /// <param name="symbol">the source symbol</param>
        /// <param name="pOut">The result name</param>
        /// <param name="success">Whether the result is contained in out.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXSymbolGetName(SymbolHandle symbol,
                              byte** pOut,
                              int* success);

        /// <summary>
        /// Get string attribute from symbol
        /// </summary>
        /// <param name="symbol">the source symbol</param>
        /// <param name="key">The key of the symbol.</param>
        /// <param name="pOut">The result attribute, can be NULL if the attribute do not exist.</param>
        /// <param name="success">Whether the result is contained in out.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXSymbolGetAttr(SymbolHandle symbol,
                              byte* key,
                              byte** pOut,
                              int* success);

        /// <summary>
        /// Set string attribute from symbol.
        /// NOTE: Setting attribute to a symbol can affect the semantics(mutable/immutable) of symbolic graph.
        /// 
        /// Safe recommendaton: use  immutable graph
        ///  - Only allow set attributes during creation of new symbol as optional parameter
        ///  
        /// Mutable graph (be careful about the semantics):
        /// - Allow set attr at any point.
        /// - Mutating an attribute of some common node of two graphs can cause confusion from user.
        /// 
        /// </summary>
        /// <param name="symbol">the source symbol</param>
        /// <param name="key">The key of the symbol.</param>
        /// <param name="value">The value to be saved.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXSymbolSetAttr(SymbolHandle symbol,
                              byte* key,
                              byte* value);

        /// <summary>
        /// Get all attributes from symbol, including all descendents.
        /// </summary>
        /// <param name="symbol">the source symbol</param>
        /// <param name="out_size">The number of output attributes</param>
        /// <param name="pOut">2*out_size strings representing key value pairs.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXSymbolListAttr(SymbolHandle symbol,
                               mx_uint* out_size,
                               byte*** pOut);

        /// <summary>
        /// Get all attributes from symbol, excluding descendents.
        /// </summary>
        /// <param name="symbol">the source symbol</param>
        /// <param name="out_size">The number of output attributes</param>
        /// <param name="pOut">2*out_size strings representing key value pairs.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXSymbolListAttrShallow(SymbolHandle symbol,
                                      mx_uint* out_size,
                                      byte*** pOut);

        /// <summary>
        /// List arguments in the symbol.
        /// </summary>
        /// <param name="symbol">the symbol</param>
        /// <param name="out_size">output size</param>
        /// <param name="out_str_array">pointer to hold the output string array</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXSymbolListArguments(SymbolHandle symbol,
                                    mx_uint* out_size,
                                    byte*** out_str_array);

        /// <summary>
        /// List returns in the symbol.
        /// </summary>
        /// <param name="symbol">the symbol</param>
        /// <param name="out_size"> output size</param>
        /// <param name="out_str_array">pointer to hold the output string array</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXSymbolListOutputs(SymbolHandle symbol,
                                  mx_uint* out_size,
                                  byte*** out_str_array);

        /// <summary>
        /// Get a symbol that contains all the internals.
        /// </summary>
        /// <param name="symbol">The symbol</param>
        /// <param name="pOut">The output symbol whose outputs are all the internals.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXSymbolGetInternals(SymbolHandle symbol,
                                   SymbolHandle* pOut);

        /// <summary>
        /// Get index-th outputs of the symbol.
        /// </summary>
        /// <param name="symbol">The symbol</param>
        /// <param name="index">the Index of the output.</param>
        /// <param name="pOut">The output symbol whose outputs are the index-th symbol.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXSymbolGetOutput(SymbolHandle symbol,
                                mx_uint index,
                                SymbolHandle* pOut);

        /// <summary>
        /// List auxiliary states in the symbol.
        /// </summary>
        /// <param name="symbol">the symbol</param>
        /// <param name="out_size">output size</param>
        /// <param name="out_str_array">pointer to hold the output string array</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXSymbolListAuxiliaryStates(SymbolHandle symbol,
                                          mx_uint* out_size,
                                          byte*** out_str_array);

        /// <summary>
        /// Compose the symbol on other symbols.
        /// 
        /// This function will change the sym hanlde.
        /// To achieve function apply behavior, copy the symbol first
        /// before apply.
        /// </summary>
        /// <param name="sym">the symbol to apply</param>
        /// <param name="name">the name of symbol</param>
        /// <param name="num_args">number of arguments</param>
        /// <param name="keys">the key of keyword args (optional)</param>
        /// <param name="args">arguments to sym</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXSymbolCompose(SymbolHandle sym,
                              byte* name,
                              mx_uint num_args,
                              byte** keys,
                              SymbolHandle* args);

        /// <summary>
        /// Get the gradient graph of the symbol
        /// </summary>
        /// <param name="sym">the symbol to get gradient</param>
        /// <param name="num_wrt">number of arguments to get gradient</param>
        /// <param name="wrt">the name of the arguments to get gradient</param>
        /// <param name="pOut">the returned symbol that has gradient</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXSymbolGrad(SymbolHandle sym,
                           mx_uint num_wrt,
                           byte** wrt,
                           SymbolHandle* pOut);

        /// <summary>
        /// infer shape of unknown input shapes given the known one.
        /// The shapes are packed into a CSR matrix represented by arg_ind_ptr and arg_shape_data
        /// The call will be treated as a kwargs call if key != nullptr or num_args==0, otherwise it is positional.
        /// </summary>
        /// <param name="sym">symbol handle</param>
        /// <param name="num_args">numbe of input arguments.</param>
        /// <param name="keys">the key of keyword args (optional)</param>
        /// <param name="arg_ind_ptr">he head pointer of the rows in CSR</param>
        /// <param name="arg_shape_data">the content of the CSR</param>
        /// <param name="in_shape_size">sizeof the returning array of in_shapes</param>
        /// <param name="in_shape_ndim">returning array of shape dimensions of eachs input shape.</param>
        /// <param name="in_shape_data">returning array of pointers to head of the input shape.</param>
        /// <param name="out_shape_size">sizeof the returning array of out_shapes</param>
        /// <param name="out_shape_ndim">returning array of shape dimensions of eachs input shape.</param>
        /// <param name="out_shape_data">returning array of pointers to head of the input shape.</param>
        /// <param name="aux_shape_size">sizeof the returning array of aux_shapes</param>
        /// <param name="aux_shape_ndim">returning array of shape dimensions of eachs auxiliary shape.</param>
        /// <param name="aux_shape_data">returning array of pointers to head of the auxiliary shape.</param>
        /// <param name="complete">whether infer shape completes or more information is needed.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXSymbolInferShape(SymbolHandle sym,
                                 mx_uint num_args,
                                 byte** keys,
                                 mx_uint* arg_ind_ptr,
                                  mx_uint* arg_shape_data,
                                 mx_uint* in_shape_size,
                                  mx_uint** in_shape_ndim,
                                  mx_uint*** in_shape_data,
                                 mx_uint* out_shape_size,
                                  mx_uint** out_shape_ndim,
                                  mx_uint*** out_shape_data,
                                 mx_uint* aux_shape_size,
                                  mx_uint** aux_shape_ndim,
                                  mx_uint*** aux_shape_data,
                                 int* complete);

        /// <summary>
        /// partially infer shape of unknown input shapes given the known one.
        /// 
        /// Return partially inferred results if not all shapes could be inferred.
        /// The shapes are packed into a CSR matrix represented by arg_ind_ptr and arg_shape_data
        /// The call will be treated as a kwargs call if key != nullptr or num_args==0, otherwise it is positional.
        /// </summary>
        /// <param name="sym">symbol handle</param>
        /// <param name="num_args">numbe of input arguments.</param>
        /// <param name="keys">the key of keyword args (optional)</param>
        /// <param name="arg_ind_ptr">the head pointer of the rows in CSR</param>
        /// <param name="arg_shape_data">the content of the CSR</param>
        /// <param name="in_shape_size">sizeof the returning array of in_shapes</param>
        /// <param name="in_shape_ndim">returning array of shape dimensions of eachs input shape.</param>
        /// <param name="in_shape_data">returning array of pointers to head of the input shape.</param>
        /// <param name="out_shape_size">sizeof the returning array of out_shapes</param>
        /// <param name="out_shape_ndim">returning array of shape dimensions of eachs input shape.</param>
        /// <param name="out_shape_data">returning array of pointers to head of the input shape.</param>
        /// <param name="aux_shape_size">sizeof the returning array of aux_shapes</param>
        /// <param name="aux_shape_ndim">returning array of shape dimensions of eachs auxiliary shape.</param>
        /// <param name="aux_shape_data">returning array of pointers to head of the auxiliary shape.</param>
        /// <param name="complete">whether infer shape completes or more information is needed.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXSymbolInferShapePartial(SymbolHandle sym,
                                 mx_uint num_args,
                                 byte** keys,
                                  mx_uint* arg_ind_ptr,
                                  mx_uint* arg_shape_data,
                                 mx_uint* in_shape_size,
                                  mx_uint** in_shape_ndim,
                                  mx_uint*** in_shape_data,
                                 mx_uint* out_shape_size,
                                  mx_uint** out_shape_ndim,
                                  mx_uint*** out_shape_data,
                                 mx_uint* aux_shape_size,
                                  mx_uint** aux_shape_ndim,
                                  mx_uint*** aux_shape_data,
                                 int* complete);

        /// <summary>
        /// infer type of unknown input types given the known one.
        /// The types are packed into a CSR matrix represented by arg_ind_ptr and arg_type_data
        /// The call will be treated as a kwargs call if key != nullptr or num_args==0, otherwise it is positional.
        /// </summary>
        /// <param name="sym">symbol handle</param>
        /// <param name="num_args">numbe of input arguments.</param>
        /// <param name="keys">the key of keyword args (optional)</param>
        /// <param name="arg_type_data">the content of the CSR</param>
        /// <param name="in_type_size">sizeof the returning array of in_types</param>
        /// <param name="in_type_data">returning array of pointers to head of the input type.</param>
        /// <param name="out_type_size">sizeof the returning array of out_types</param>
        /// <param name="out_type_data">returning array of pointers to head of the input type.</param>
        /// <param name="aux_type_size">sizeof the returning array of aux_types</param>
        /// <param name="aux_type_data">returning array of pointers to head of the auxiliary type.</param>
        /// <param name="complete">whether infer type completes or more information is needed.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXSymbolInferType(SymbolHandle sym,
                                mx_uint num_args,
                                Byte** keys,
                                 int* arg_type_data,
                                mx_uint* in_type_size,
                                 int** in_type_data,
                                mx_uint* out_type_size,
                                 int** out_type_data,
                                mx_uint* aux_type_size,
                                 int** aux_type_data,
                                int* complete);


        /// <summary>
        /// list all the available operator names, include entries.
        /// </summary>
        /// <param name="out_size">the size of returned array</param>
        /// <param name="out_array">the output operator name array.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int NNListAllOpNames(nn_uint* out_size,
                              Byte*** out_array);


        /// <summary>
        /// Get operator handle given name.
        /// </summary>
        /// <param name="op_name">The name of the operator.</param>
        /// <param name="op_out">The returnning op handle.</param>
        /// <returns></returns>
        [DllImport(MXNET_DLL)]
        public static extern int NNGetOpHandle(byte* op_name,
                           OpHandle* op_out);

        #endregion

        #region Executor interface

        /// <summary>
        /// Delete the executor
        /// </summary>
        /// <param name="handle">the executor</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXExecutorFree(ExecutorHandle handle);

        /// <summary>
        /// Print the content of execution plan, used for debug.
        /// </summary>
        /// <param name="handle">handle the executor.</param>
        /// <param name="out_str">pointer to hold the output string of the printing.</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXExecutorPrint(ExecutorHandle handle, byte** out_str);

        /// <summary>
        /// Executor forward method
        /// </summary>
        /// <param name="handle">handle executor handle</param>
        /// <param name="is_train">int value to indicate whether the forward pass is for evaluation</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXExecutorForward(ExecutorHandle handle, int is_train);

        /// <summary>
        /// Excecutor run backward
        /// </summary>
        /// <param name="handle">handle execute handle</param>
        /// <param name="len">lenth</param>
        /// <param name="head_grads">NDArray handle for heads' gradient</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXExecutorBackward(ExecutorHandle handle,
                                 mx_uint len,
                                 NDArrayHandle* head_grads);


        /// <summary>
        /// Get executor's head NDArray
        /// </summary>
        /// <param name="handle">executor handle</param>
        /// <param name="out_size">output narray vector size</param>
        /// <param name="pOut">out put narray handles</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXExecutorOutputs(ExecutorHandle handle,
                                mx_uint* out_size,
                                NDArrayHandle** pOut);

        /// <summary>
        /// Generate Executor from symbol
        /// </summary>
        /// <param name="symbol_handle">symbol handle</param>
        /// <param name="dev_type">device type</param>
        /// <param name="dev_id">device id</param>
        /// <param name="len">length</param>
        /// <param name="in_args">in args array</param>
        /// <param name="arg_grad_store">arg grads handle array</param>
        /// <param name="grad_req_type">grad req array</param>
        /// <param name="aux_states_len">length of auxiliary states</param>
        /// <param name="aux_states">auxiliary states array</param>
        /// <param name="pOut">output executor handle</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXExecutorBind(SymbolHandle symbol_handle,
                             int dev_type,
                             int dev_id,
                             mx_uint len,
                             NDArrayHandle* in_args,
                             NDArrayHandle* arg_grad_store,
                             mx_uint* grad_req_type,
                             mx_uint aux_states_len,
                             NDArrayHandle* aux_states,
                             ExecutorHandle* pOut);

        /// <summary>
        /// Generate Executor from symbol,
        /// This is advanced function, allow specify group2ctx map.
        /// The user can annotate "ctx_group" attribute to name each group.
        /// </summary>
        /// <param name="symbol_handle">symbol handle</param>
        /// <param name="dev_type">device type of default context</param>
        /// <param name="dev_id">device id of default context</param>
        /// <param name="num_map_keys">size of group2ctx map</param>
        /// <param name="map_keys">keys of group2ctx map</param>
        /// <param name="map_dev_types">device type of group2ctx map</param>
        /// <param name="map_dev_ids">device id of group2ctx map</param>
        /// <param name="len">length</param>
        /// <param name="in_args">in args array</param>
        /// <param name="arg_grad_store">arg grads handle array</param>
        /// <param name="grad_req_type">grad req array</param>
        /// <param name="aux_states_len">length of auxiliary states</param>
        /// <param name="aux_states">auxiliary states array</param>
        /// <param name="pOut">output executor handle</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXExecutorBindX(SymbolHandle symbol_handle,
                              int dev_type,
                              int dev_id,
                              mx_uint num_map_keys,
                              Byte** map_keys,
                               int* map_dev_types,
                               int* map_dev_ids,
                              mx_uint len,
                              NDArrayHandle* in_args,
                              NDArrayHandle *arg_grad_store,
                              mx_uint* grad_req_type,
                              mx_uint aux_states_len,
                              NDArrayHandle* aux_states,
                              ExecutorHandle * pOut);

        /// <summary>
        ///  Generate Executor from symbol,
        ///  This is advanced function, allow specify group2ctx map.
        ///  The user can annotate "ctx_group" attribute to name each group.
        /// </summary>
        /// <param name="symbol_handle">symbol handle</param>
        /// <param name="dev_type">device type of default context</param>
        /// <param name="dev_id">device id of default context</param>
        /// <param name="num_map_keys">size of group2ctx map</param>
        /// <param name="map_keys">keys of group2ctx map</param>
        /// <param name="map_dev_types">device type of group2ctx map</param>
        /// <param name="map_dev_ids">device id of group2ctx map</param>
        /// <param name="len">length</param>
        /// <param name="in_args">in args array</param>
        /// <param name="arg_grad_store">arg grads handle array</param>
        /// <param name="grad_req_type">grad req array</param>
        /// <param name="aux_states_len">length of auxiliary states</param>
        /// <param name="aux_states">auxiliary states array</param>
        /// <param name="shared_exec">input executor handle for memory sharing</param>
        /// <param name="pOut">output executor handle</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXExecutorBindEX(SymbolHandle symbol_handle,
                               int dev_type,
                               int dev_id,
                               mx_uint num_map_keys,
                               byte** map_keys,
                                int* map_dev_types,
                                int* map_dev_ids,
                               mx_uint len,
                               NDArrayHandle* in_args,
                               NDArrayHandle *arg_grad_store,
                               mx_uint* grad_req_type,
                               mx_uint aux_states_len,
                               NDArrayHandle* aux_states,
                               ExecutorHandle shared_exec,
                               ExecutorHandle* pOut);


        /// <summary>
        /// set a call back to notify the completion of operation
        /// </summary>
        /// <param name="handle"></param>
        /// <param name="callback"></param>
        /// <param name="callback_handle"></param>
        /// <returns></returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXExecutorSetMonitorCallback(ExecutorHandle handle,
                                           ExecutorMonitorCallback callback,
                                           void* callback_handle);

        #endregion

        #region CustomOp

        public struct CustomOpInfo
        {
            public void* forward;
            public void* backward;
            public void* del;
            public void* p_forward;
            public void* p_backward;
            public void* p_del;
        }

        public struct CustomOpPropInfo
        {
            public void* list_arguments;
            public void* list_outputs;
            public void* infer_shape;
            public void* declare_backward_dependency;
            public void* create_operator;
            public void* list_auxiliary_states;
            public void* del;

            public void* p_list_arguments;
            public void* p_list_outputs;
            public void* p_infer_shape;
            public void* p_declare_backward_dependency;
            public void* p_create_operator;
            public void* p_list_auxiliary_states;
            public void* p_del;
        }

        public delegate bool CustomOpPropCreator(String opType, int numKwargs, byte** keys, byte** values, CustomOpPropInfo* ret);

        [DllImport(MXNET_DLL)]
        public static extern int MXCustomOpRegister(String opType, void* creator);

        public static int MXCustomOpRegister(String opType, CustomOpPropCreator creator)
        {
            GCHandle handle = GCHandle.Alloc(creator);
            GCHandleManager.Instance.Add(handle);

            IntPtr ptr = Marshal.GetFunctionPointerForDelegate(creator);
            return MXCustomOpRegister(opType, (void*)ptr);
        }

        #endregion
    }

    public enum OpReqType
    {
        /*! \brief no operation, do not write anything */
        kNullOp,
        /*! \brief write gradient to provided space */
        kWriteTo,
        /*!
        * \brief perform an inplace write,
        * Target shares memory with one of input arguments.
        * This option only happen when
        */
        kWriteInplace,
        /*! \brief add to the provided space */
        kAddTo
    };
}
