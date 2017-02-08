using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Linq;
using System.Threading.Tasks;

namespace MXNetSharp
{
    public unsafe class CustomOpExecutor
    {
        public bool Forward(int size, void** ptrs, int* tags, int* reqs, bool isTrain, void* state)
        {
            throw new NotImplementedException();
        }

        public bool Backward(int size, void** ptrs, int* tags, int* reqs, bool isTrain, void* state)
        {
            throw new NotImplementedException();
        }

        public bool Del(void* state)
        {
            throw new NotImplementedException();
        }

    }

    public unsafe class CustomOp
    {
        #region delegates
        public delegate bool __ListArguments(byte*** args, void* state);
        public delegate bool __ListOutputs(Byte*** outputs, void* state);
        public delegate bool __InferShape(int numInput, int* ndims, uint** shapes, void* state);
        public delegate bool __DeclareBackwardDependency(int* outGrad, int* inData, int* outData, int* numDeps, int** rdeps, void* state);
        public delegate bool __CreateOperator(byte* ctx, int numInputs, uint** shapes, int* ndims, int* dtypes, CAPI.CustomOpInfo* ret, void* state);
        public delegate bool __ListAuxiliaryStates(byte*** aux, void* state);
        public delegate bool __Forward(int size, void** ptrs, int* tags, int* reqs, bool isTrain, void* state);
        public delegate bool __Backward(int size, void** ptrs, int* tags, int* reqs, bool isTrain, void* state);
        public delegate bool __Del(void* state);
        #endregion

        protected CustomOpExecutor _opExecutor;

        public bool ListArguments(byte*** args, void* state)
        {
            throw new NotImplementedException();
        }

        public List<String> ListArguments()
        {
            return null;
        }

        public bool ListOutputs(Byte*** outputs, void* state)
        {
            throw new NotImplementedException();
        }

        public List<String> ListOutputs()
        {
            return null;
        }

        public bool ListAuxiliaryStates(byte*** aux, void* state)
        {
            throw new NotImplementedException();
        }

        public List<String> ListAuxiliaryStates()
        {
            return null;
        }

        public bool DelOp(void* state)
        {
            throw new NotImplementedException();
        }

        public bool InferShape(int numInput, int* ndims, uint** shapes, void* state)
        {
            throw new NotImplementedException();
        }

        public Tuple<Shape,Shape,Shape> InferShape(Shape inShape)
        {
            throw new NotImplementedException();
        }

        public bool DeclareBackwardDependency(int* outGrad, int* inData, int* outData, int* numDeps, int** rdeps, void* state)
        {
            throw new NotImplementedException();
        }

        public bool CreateOperator(byte* ctx, int numInputs, uint** shapes, int* ndims, int* dtypes, CAPI.CustomOpInfo* ret, void* state)
        {
            ret->forward = GetFunctionPointer(new __Forward(this._opExecutor.Forward));
            ret->backward = GetFunctionPointer(new __Backward(this._opExecutor.Backward));
            ret->del = GetFunctionPointer(new __Del(this._opExecutor.Del));

            throw new NotImplementedException();
        }

        private static void* GetFunctionPointer<T>(T func)
        {
            GCHandle handle = GCHandle.Alloc(func);
            GCHandleManager.Instance.Add(handle);
            IntPtr ptr = Marshal.GetFunctionPointerForDelegate(func);
            return (void*)ptr;
        }

        public bool CreateCustomOpProp(String opType, int numKwargs, byte** keys, byte** values, CAPI.CustomOpPropInfo* ret)
        {
            ret->list_arguments = GetFunctionPointer(new __ListArguments(this.ListArguments));
            ret->list_outputs = GetFunctionPointer(new __ListOutputs(this.ListOutputs));
            ret->infer_shape = GetFunctionPointer(new __InferShape(this.InferShape));
            ret->declare_backward_dependency = GetFunctionPointer(new __DeclareBackwardDependency(this.DeclareBackwardDependency));
            ret->create_operator = GetFunctionPointer(new __CreateOperator(this.CreateOperator));
            ret->list_auxiliary_states = GetFunctionPointer(new __ListAuxiliaryStates(this.ListAuxiliaryStates));
            ret->del = GetFunctionPointer(new __Del(this.DelOp));
            ret->p_create_operator = null;
            ret->p_declare_backward_dependency = null;
            ret->p_del = null;
            ret->p_infer_shape = null;
            ret->p_list_arguments = null;
            ret->p_list_auxiliary_states = null;
            ret->p_list_outputs = null;

            return false;
        }

        public void Register(String opType)
        {
            CAPI.CustomOpPropCreator creator = new CAPI.CustomOpPropCreator(this.CreateCustomOpProp);
            int rtn = CAPI.MXCustomOpRegister(opType, creator);
        }
    }
}
