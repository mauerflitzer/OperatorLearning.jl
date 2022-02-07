"""
`DeepONet(architecture_branch::Tuple, architecture_trunk::Tuple,
                        act_branch = identity, act_trunk = identity;
                        init_branch = Flux.glorot_uniform,
                        init_trunk = Flux.glorot_uniform,
                        bias_branch=true, bias_trunk=true)`
`DeepONet(branch_net::Flux.Chain, trunk_net::Flux.Chain)`

Create an (unstacked) DeepONet architecture as proposed by Lu et al.
arXiv:1910.03193

The model works as follows:

x --- branch --
               |
                -⊠--u-
               |
y --- trunk ---

Where `x` represent the parameters of the PDE, discretely evaluated at its respective sensors,
and `y` are the probing locations for the operator to be trained.
`u` is the solution of the queried instance of the PDE, given by the specific choice of parameters.

Both inputs `x` and `y` are multiplied together via dot product Σᵢ bᵢⱼ tᵢₖ.

You can set up this architecture in two ways:

1. By Specifying the architecture and all its parameters as given above. This always creates `Dense` layers for the branch and trunk net and corresponds to the DeepONet proposed by Lu et al.

2. By passing two architectures in the form of two Chain structs directly. Do this if you want more flexibility and e.g. use an RNN or CNN instead of simple `Dense` layers.

Strictly speaking, DeepONet does not imply either of the branch or trunk net to be a simple DNN. Usually though, this is the case which is why it's treated as the default case here.

```julia
julia> model = DeepONet((32,64,72), (24,64,72))
DeepONet with
branch net: (Chain(Dense(32, 64), Dense(64, 72)))
Trunk net: (Chain(Dense(24, 64), Dense(64, 72)))

julia> model = DeepONet((32,64,72), (24,64,72), σ, tanh; init_branch=Flux.glorot_normal, bias_trunk=false)
DeepONet with
branch net: (Chain(Dense(32, 64, σ), Dense(64, 72, σ)))
Trunk net: (Chain(Dense(24, 64, tanh; bias=false), Dense(64, 72, tanh; bias=false)))

julia> branch = Chain(Dense(2,128),Dense(128,64),Dense(64,72))
Chain(
  Dense(2, 128),                        # 384 parameters
  Dense(128, 64),                       # 8_256 parameters
  Dense(64, 72),                        # 4_680 parameters
)                   # Total: 6 arrays, 13_320 parameters, 52.406 KiB.

julia> trunk = Chain(Dense(1,24),Dense(24,72))
Chain(
  Dense(1, 24),                         # 48 parameters
  Dense(24, 72),                        # 1_800 parameters
)                   # Total: 4 arrays, 1_848 parameters, 7.469 KiB.

julia> model = DeepONet(branch,trunk)
DeepONet with
branch net: (Chain(Dense(2, 128), Dense(128, 64), Dense(64, 72)))
Trunk net: (Chain(Dense(1, 24), Dense(24, 72)))
```
"""
struct DeepONet
    branch_net::Flux.Chain
    trunk_net::Flux.Chain
    # Constructor for the DeepONet
    function DeepONet(
        branch_net::Flux.Chain,
        trunk_net::Flux.Chain)
        new(branch_net, trunk_net)
    end
end

# Declare the function that assigns Weights and biases to the layer
function DeepONet(architecture_branch::Tuple, architecture_trunk::Tuple,
                        act_branch = identity, act_trunk = identity;
                        init_branch = Flux.glorot_uniform,
                        init_trunk = Flux.glorot_uniform,
                        bias_branch=true, bias_trunk=true)

    # To construct the subnets we use the helper function in subnets.jl
    # Initialize the branch net
    branch_net = construct_subnet(architecture_branch, act_branch;
                                    init=init_branch, bias=bias_branch)
    # Initialize the trunk net
    trunk_net = construct_subnet(architecture_trunk, act_trunk;
                                    init=init_trunk, bias=bias_trunk)

    return DeepONet(branch_net, trunk_net)
end

Flux.@functor DeepONet

# The actual layer that does stuff
# x needs to be at least a 2-dim array,
# since we need n inputs, evaluated at m locations
function (a::DeepONet)(x::AbstractMatrix, y::AbstractVecOrMat)
    # Assign the parameters
    branch, trunk = a.branch_net, a.trunk_net

    # Dot product needs a dim to contract
    # However, inputs are normally given with batching done in the same dim
    # so we need to adjust (i.e. transpose) one of the inputs,
    # and that's easiest on the matrix-type input
    return branch(x) * trunk(y)'
end

# Handling batches:
# We use basically the same function, but using NNlib's batched_mul instead of
# regular matrix-matrix multiplication
function (a::DeepONet)(x::AbstractArray, y::AbstractVecOrMat)
    # Assign the parameters
    branch, trunk = a.branch_net, a.trunk_net

    # Dot product needs a dim to contract
    # However, inputs are normally given with batching done in the same dim
    # so we need to adjust (i.e. transpose) one of the inputs,
    # and that's easiest on the matrix-type input
    return branch(x) ⊠ trunk(y)'
end

# Sensors stay the same and shouldn't be batched
(a::DeepONet)(x::AbstractArray, y::AbstractArray) = 
  throw(ArgumentError("Sensor locations fed to trunk net can't be batched."))

# Print nicely
function Base.show(io::IO, l::DeepONet)
    print(io, "DeepONet with\nbranch net: (",l.branch_net)
    print(io, ")\n")
    print(io, "Trunk net: (", l.trunk_net)
    print(io, ")\n")
end