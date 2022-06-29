using Flux: length, reshape, train!, throttle, @epochs, params
using Zygote: jacobian, gradient, hessian
using OperatorLearning, Flux, MAT
using Zygote
using GalacticOptim, GalacticOptimJL
using ModelingToolkit
using ModelingToolkit: Interval
using Dates
using LinearAlgebra
using OMEinsum


device = gpu;


#TODO modelling:
# @parameters t, x
# @variables u(..)
# Dt = Differential(t)
# Dx = Differential(x)
# Dxx = Differential(x)^2

# eq = Dt(u(t, x)) + u(t, x) * Dx(u(t, x)) - Dxx(u(t, x)) ~ 0

# # Initial and boundary conditions
# bcs = [u(0, x) ~ -sin(pi * x),
#     u(t, 0) ~ 0.0,
#     u(t, 1) ~ 0.0,
#     u(t, 0) ~ u(t, 0)]

# # Space and time domains
# domains = [t ∈ Interval(0.0, 1.0),
#     x ∈ Interval(0.0, 1.0)]

#=
We would like to implement and train a DeepONet that infers the solution
u(x) of the burgers equation on a grid of 1024 points at time one based
on the initial condition a(x) = u(x,0)
=#

# Read the data from MAT file and store it in a dict
# key "a" is the IC
# key "u" is the desired solution at time 1
vars = matread("burgers_data_R10.mat") |> device

# For trial purposes, we might want to train with different resolutions
# So we sample only every n-th element
subsample = 2^7;
batch_size = 10
# create the x training array, according to our desired grid size
xtrain = vars["a"][1:batch_size, 1:subsample:end]' |> device;
# create the x test array
xtest = vars["a"][end-batch_size:end, 1:subsample:end]' |> device;

# Create the y training array
ytrain = vars["u"][1:batch_size, 1:subsample:end] |> device;
# Create the y test array
ytest = vars["u"][end-batch_size:end, 1:subsample:end] |> device;

# The data is missing grid data, so we create it
# `collect` converts data type `range` into an array
# vcat stacks the 2 grid vectors
sensor_count = 64
x_grid = collect(range(0, 1, length=sensor_count))'
t_grid = ones(1,sensor_count)
grid = vcat(x_grid, t_grid) |> device
# Create the DeepONet:
# IC is given on grid of 1024 points, and we solve for a fixed time t in one
# spatial dimension x, making the branch input of size 1024 and trunk size 1
# We choose GeLU activation for both subnets
model = DeepONet((sensor_count, sensor_count, sensor_count), (2, sensor_count, sensor_count), gelu, gelu) |> device

# We use the ADAM optimizer for training
learning_rate = 0.001
opt = ADAM(learning_rate)

# Specify the model parameters
parameters = params(model)

# The loss function
# We can't use the "vanilla" implementation of the mse here since we have
# two distinct inputs to our DeepONet, so we wrap them into a tuple


function modell_call(x, x_grid, t_grid)
    model(x, vcat(x_grid, t_grid))
end



function loss_neu(xtrain, x_sensor, t_sensor, ytrain)
    f = modell_call(xtrain, x_sensor, t_sensor)
    #print("Before: $(Dates.format(now(), "HH:MM:SS"))\n")
    f_t = reshape(jacobian(t_grid->modell_call(xtrain,x_sensor,t_sensor), t_grid)[1], (batch_size, sensor_count, sensor_count))
    f_x = reshape(jacobian(x_grid->modell_call(xtrain,x_sensor,t_sensor), x_grid)[1], (batch_size, sensor_count, sensor_count))
    #print("Jacobians done: $(Dates.format(now(), "HH:MM:SS"))\n")
    f_xx_diag = Float64[]
    for j in 1:batch_size
        for i in 1:sensor_count
            push!(f_xx_diag,diaghessian(x_sensor->modell_call(xtrain, x_sensor, t_grid)[i], x_grid)[1][i])
        end
    end
    f_xx_diag = reshape(f_xx_diag,(batch_size, sensor_count))
    f_t = mapslices(diag, f_t, dims=[2,3])
    f_x = mapslices(diag, f_x, dims=[2,3])
    #print("After: $(Dates.format(now(), "HH:MM:SS"))\n")
    @ein out[batch, x] := f[batch, x] * f_x[batch, x, 1]
    out = out + f_t - f_xx_diag
    return 20* Flux.Losses.mse(zeros(batch_size, sensor_count), out) + Flux.Losses.mse(model(xtrain, sensor), ytrain)
end




loss(xtrain, ytrain, sensor) = Flux.Losses.mse(model(xtrain, sensor), ytrain)

# Define a callback function that gives some output during training
evalcb() = @show(loss(xtest, ytest, grid))
# Print the callback only every 5 seconds
throttled_cb = throttle(evalcb, 5)
#NeuralPDE workflow
# res = GalacticOptim.solve(prob, opt, progress=false; cb=evalcb, maxiters=100)

# Do the training loop
#Flux.@epochs 500 train!(loss, parameters, [(xtrain, ytrain, grid)], opt, cb=evalcb)
Flux.@epochs 10 train!(loss_neu, parameters, [(xtrain, x_grid, t_grid, ytrain)], opt, cb=evalcb)
