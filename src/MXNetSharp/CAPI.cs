using System;
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

    public unsafe class CAPI
    {
        public const String MXNET_DLL = "libmxnet";

        public delegate IntPtr ExecutorMonitorCallback(Byte* pChars, NDArrayHandle handle, void * p);

        [DllImport(MXNET_DLL)]
        public static extern String MXGetLastError();

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
                              NDArrayHandle *pOut);

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
                              NDArrayHandle *pOut);

        /// <summary>
        /// free the narray handle
        /// </summary>
        /// <param name="handle">the handle to be freed</param>
        /// <returns>0 when success, -1 when failure happens</returns>
        [DllImport(MXNET_DLL)]
        public static extern int MXNDArrayFree(NDArrayHandle handle);

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
                                       uint size);

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
