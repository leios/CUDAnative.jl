# contextual dispatch using Cassette.jl
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

@inline unknowably_false() = Base.llvmcall("ret i8 0", Bool, Tuple{})

function generate_transform(method_redefinitions)
    return function transform(ctx, ref)
        CI = ref.code_info

        # inline everything
        noinline = any(@nospecialize(x) ->
                           Core.Compiler.isexpr(x, :meta) &&
                           x.args[1] == :noinline,
                       CI.code)
        CI.inlineable = !noinline

        if method_redefinitions
            # 265 fix, insert a call to the original method
            # that we later will remove with LLVM's DCE
            # TODO: We also don't want to compile these functions
            unknowably_false = GlobalRef(@__MODULE__, :unknowably_false)
            Cassette.insert_statements!(CI.code, CI.codelocs,
              (x, i) -> i == 1 ?  4 : nothing,
              (x, i) -> i == 1 ? [
                  Expr(:call, Expr(:nooverdub, unknowably_false)),
                  Expr(:gotoifnot, Core.SSAValue(i), i+3),
                  Expr(:call, Expr(:nooverdub, Core.SlotNumber(1)), (Core.SlotNumber(i) for i in 2:ref.method.nargs)...),
                  x] : nothing)
        end
        CI.ssavaluetypes = length(CI.code)

        #Core.Compiler.validate_code(CI)
        return CI
    end
end

const StaticPass = Cassette.@pass generate_transform(false)
const InteractivePass = Cassette.@pass generate_transform(true)

Cassette.@context CUDACtx
const StaticCtx = Cassette.disablehooks(CUDACtx(pass = StaticPass))
const InteractiveCtx = Cassette.disablehooks(CUDACtx(pass = InteractivePass))

@inline function contextualize(f::F, interactive) where F
    ctx = interactive ? InteractiveCtx : StaticCtx
    (args...) -> Cassette.overdub(ctx, f, args...)
end

###
# Cassette fixes
###

# kwfunc fix
Cassette.overdub(::CUDACtx, ::typeof(Core.kwfunc), f) = return Core.kwfunc(f)

# the functions below are marked `@pure` and by rewritting them we hide that from
# inference so we leave them alone (see https://github.com/jrevels/Cassette.jl/issues/108).
@inline Cassette.overdub(::CUDACtx, ::typeof(Base.isimmutable), x)     = return Base.isimmutable(x)
@inline Cassette.overdub(::CUDACtx, ::typeof(Base.isstructtype), t)    = return Base.isstructtype(t)
@inline Cassette.overdub(::CUDACtx, ::typeof(Base.isprimitivetype), t) = return Base.isprimitivetype(t)
@inline Cassette.overdub(::CUDACtx, ::typeof(Base.isbitstype), t)      = return Base.isbitstype(t)
@inline Cassette.overdub(::CUDACtx, ::typeof(Base.isbits), x)          = return Base.isbits(x)

@inline Cassette.overdub(::CUDACtx, ::typeof(datatype_align), ::Type{T}) where {T} = datatype_align(T) 

###
# Rewrite functions
###
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
