* Document environment variables (`CUDA_FORCE_API_VERSION`,
  `CUDA_FORCE_GPU_TRIPLE`, `CUDA_FORCE_GPU_ARCH`)

* Use a `DevicePtr` containing a `Ptr` (cfr. `CppPtr` in
  [Cxx.jl](https://github.com/Keno/Cxx.jl/blob/master/src/Cxx.jl))?

* Check if we can ditch the boxing logic, and use Julia's new `Ref` type instead

* Pass the native codegen context into `@cuda`, allowing for multiple active
  contexts.

* Related to the point above, ff we have a single `main`, instantiating `cgctx`
  and launching a kernel (a common use-case, right?) the macro expansion happens
  before the `cgctx` instantiation is executed, but the compiler _does_ use the
  codegen context internally!

* Improve error reporting when using undefined functions. Currently, Julia just
  generates valid code, calling back into the compiler (in order to support
  calling functions defined later on). This causes the PTX back-end to generate
  a `cannot box` error, while normally Julia would report `undefined function`
  at run-time.