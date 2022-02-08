using Flux: length, reshape, train!, throttle, @epochs
using OperatorLearning, Flux, MAT

device = cpu;

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
subsample = 2^3;

# create the x training array, according to our desired grid size
xtrain = vars["a"][1:1000, 1:subsample:end]' |> device;
# create the x test array
xtest = vars["a"][end-99:end, 1:subsample:end]' |> device;

# Create the y training array
ytrain = vars["u"][1:1000, 1:subsample:end] |> device;
# Create the y test array
ytest = vars["u"][end-99:end, 1:subsample:end] |> device;

# The data is missing grid data, so we create it
# `collect` converts data type `range` into an array
grid = collect(range(0, 1, length=1024))' |> device

# Pass the data to the Flux DataLoader and give it a batch of 20
#train_loader = Flux.Data.DataLoader((xtrain, ytrain), batchsize=20, shuffle=true) |> device
#test_loader = Flux.Data.DataLoader((xtest, ytest), batchsize=20, shuffle=false) |> device

# Create the DeepONet:
# IC is given on grid of 1024 points, and we solve for a fixed time t in one
# spatial dimension x, making the branch input of size 1024 and trunk size 1
model = DeepONet((1024,1024,1024),(1,1024,1024))

# We use the ADAM optimizer for training
learning_rate = 0.001
opt = ADAM(learning_rate)

# Specify the model parameters
parameters = params(model)

# The loss function
# We can't use the "vanilla" implementation of the mse here since we have
# two distinct inputs to our DeepONet, so we wrap them into a tuple
loss(xtrain,ytrain,sensor) = Flux.Losses.mse(model(xtrain,sensor),ytrain)

# Define a callback function that gives some output during training
evalcb() = @show(loss(xtest,ytest,grid))
# Print the callback only every 5 seconds
throttled_cb = throttle(evalcb, 5)

# Do the training loop
Flux.@epochs 500 train!(loss, parameters, [(xtrain,ytrain,grid)], opt, cb = evalcb)

# Accuracy metrics
val_loader = Flux.Data.DataLoader((xtest, ytest), batchsize=1, shuffle=false) |> device
loss = 0.0 |> device

for (x,y) in val_loader
    ŷ = model(x)
    loss += Flux.Losses.mse(ŷ,y)
end
