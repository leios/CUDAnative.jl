##
# Implements contextual dispatch through Cassette.jl
# Goals:
# - Rewrite common CPU functions to appropriate GPU intrinsics
# 
# TODO:
# - error (erf, ...)
# - pow
# - min, max
# - mod, rem
# - gamma
# - bessel
# - distributions
# - unsorted

using Cassette

function transform(ctx, ref)
    ci = ref.code_info
    ci.inlineable = true
    return ci
end
const InlinePass = Cassette.@pass transform

Cassette.@context CUDACtx
const cudactx = Cassette.disablehooks(CUDACtx(pass = InlinePass))

Cassette.overdub(::CUDACtx, ::typeof(datatype_align), ::Type{T}) where {T} = datatype_align(T) 
Cassette.overdub(ctx::CUDACtx, ::typeof(isdevice)) = true

# libdevice.jl
for f in (:cos, :cospi, :sin, :sinpi, :tan,
          :acos, :asin, :atan, 
          :cosh, :sinh, :tanh,
          :acosh, :asinh, :atanh, 
          :log, :log10, :log1p, :log2,
          :exp, :exp2, :exp10, :expm1, :ldexp,
          :isfinite, :isinf, :isnan,
          :signbit, :abs,
          :sqrt, :cbrt,
          :ceil, :floor,)
    @eval function Cassette.overdub(ctx::CUDACtx, ::typeof(Base.$f), x::Union{Float32, Float64})
        @Base._inline_meta
        return CUDAnative.$f(x)
    end
end

contextualize(f::F) where F = (args...) -> Cassette.overdub(cudactx, f, args...)

