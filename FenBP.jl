using Statistics
using MLDatasets
using Flux, Flux.Optimise
using Flux: onehotbatch, onecold
using Base.Iterators: partition
using ChainRulesCore
using Random; Random.seed!(43)
using CUDA
using Zygote
using Zygote: @adjoint, broadcasted, Numeric, _pullback

########################################################################
# dataloader

function loadMNIST(batchsize, flatten_data)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

    # Loading Dataset	
    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
    xtest, ytest = MLDatasets.MNIST.testdata(Float32)
	
    if flatten_data==true # Flatten data for MLP experiments
        xtrain = Flux.flatten(xtrain)
        xtest = Flux.flatten(xtest)
    else # Add singleton dimension for CNN mode
        xtrain = reshape(xtrain, 28, 28, 1, :)
        xtest = reshape(xtest, 28, 28, 1, :)
    end

    # One-hot-encode the labels
    ytrain, ytest = Flux.onehotbatch(ytrain, 0:9), Flux.onehotbatch(ytest, 0:9)

    # Create DataLoaders (mini-batch iterators)
    trainloader = Flux.DataLoader((xtrain, ytrain), batchsize=batchsize, shuffle=true, partial=false)
    testloader = Flux.DataLoader((xtest, ytest), batchsize=batchsize, partial=false)

    return (trainloader, testloader)
end

########################################################################
# Custom activations and adjoints

my_relu(x) = relu(x)

function ChainRulesCore.rrule(::typeof(my_relu), x::AbstractArray)
    z = my_relu.(x)
    function pullback(Δy)
        let kappa = 1.f0 / 256, kappa_factor = 1 / (1 - sqrt(kappa)), dy = Δy
            tauB = 1.f0 / 32; 

            condA = (x .< 0).*(dy .< -1.f-3)
            condB = .!condA

            t = kappa_factor * x
            tauA = t ./ dy
            zzA = condA.*(x .- t)
            zzB = condB.*(x .- tauB.*dy)
            tau = condA.*tauA .+ condB.*tauB
            zz = my_relu.(condA.*zzA .+ condB.*zzB)

            return NoTangent(), (z .- zz) ./ tau
        end
    end
    return z, pullback
end

@adjoint function broadcasted(::typeof(my_relu), x::Numeric)
    _pullback(my_relu, x)
end

my_hardtanh(x) = hardtanh(64.f0*x)

function ChainRulesCore.rrule(::typeof(my_hardtanh), x::AbstractArray)
    x .*= 64.0f0
    z = hardtanh.(x)
    function pullback(Δy)
        let kappa = 1.f0 / 1024, kappa_factor = 1 / (1 - sqrt(kappa))
            tauA = 1.f0 / 32
            zzA = x .- tauA.*Δy

            condB = (x .< -1).*(Δy .< -1.f-3)
            t = (x .+ 1) * kappa_factor # = (x + 1) * (1 / (1 - √η) = (x + 1) / (1 - √η)
            tauB = t ./ Δy # (x + 1) / (g*(1 - √η))
            zzB  = x .- t # ẑ
            
            condC = (x .> 1).*(Δy .> 1.f-3)
            t = (x .- 1) * kappa_factor # = (x - 1) * (1 / (1 - √η) = (x - 1) / (1 - √η)
            tauC = t ./ Δy # (x - 1) / (g*(1 - √η))
            zzC  = x .- t # ẑ

            condA = .!condB.*.!condC
            tau = condA.*tauA .+ condB.*tauB .+ condC.*tauC
            zz = hardtanh.(condA.*zzA .+ condB.*zzB .+ condC.*zzC)
            
            return NoTangent(), (z .- zz) ./ tau
        end
    end
    return z, pullback
end

@adjoint function broadcasted(::typeof(my_hardtanh), x::Numeric)
    _pullback(my_hardtanh, x)
end

########################################################################
# Utility functions

round6(x) = round(x, digits=6)
round2(x) = round(x, digits=2)

num_params(model) = sum(length, Flux.params(model)) 

function getdevice(use_CUDA)
    if CUDA.functional() && use_CUDA
        device = gpu
        @info """Checking hardware
        CPU: $(Sys.cpu_info()[1].model)
        GPU: $(CUDA.name(CUDA.device()))
        Training on GPU 🚀"""
    else
        device = cpu
        @info """Checking hardware
        CPU: $(Sys.cpu_info()[1].model)
        GPU: no CUDA capable GPU selected
        Training on CPU 🐢"""
    end
    return device
end

glorot_neg_uniform(rng::AbstractRNG, dims...) = -abs.((rand(rng, Float32, dims...) .- 0.5f0) .* sqrt(24.0f0 / sum(Flux.nfan(dims...))))
glorot_neg_uniform(dims...) = glorot_neg_uniform(Random.GLOBAL_RNG, dims...)
glorot_neg_uniform(rng::AbstractRNG) = (dims...) -> glorot_neg_uniform(rng, dims...)

loss(y, ŷ) = Flux.Losses.logitcrossentropy(y, ŷ, agg=sum)

# function loss_and_accuracy(data_loader, net, device)
#     acc = 0.0f0
#     ls = 0.0f0
#     num = 0
#     for (x, y) in data_loader
#         ŷ = net(x)
#         y, ŷ = cpu(y), cpu(ŷ)
#         ls += Flux.Losses.logitcrossentropy(ŷ, y, agg=sum)
#         acc += sum(onecold(ŷ) .== onecold(y))
#         num +=  size(y, 2)
#     end
#     return ls / num, acc / num
# end

function loss_and_accuracy(data_loader, net, device)
    acc = 0
    ls = 0.0f0
    num = 0
    for (x, y) in data_loader
        x, y = x |> device, y |> device
        ls += loss(net(x), y)
        acc += sum(onecold(cpu(net(x))) .== onecold(cpu(y)))
        num +=  size(x, 2)
    end
    return ls / num, acc / num
end

function LeNet5(;act, init, imgsize=(28,28,1), nclasses=10) 
    out_conv_size = (imgsize[1]÷4 - 3, imgsize[2]÷4 - 3, 16)
    
    return Chain(
            Conv((5, 5), imgsize[end]=>6, act, init=init),
            MaxPool((2, 2)),
            Conv((5, 5), 6=>16, act, init=init),
            MaxPool((2, 2)),
            flatten,
            Dense(prod(out_conv_size), 120, act, init=init), 
            Dense(120, 84, act), 
            Dense(84, nclasses)
          )
end

function MLP(;act, layerwidth, init)
    return Chain(Dense(784, layerwidth, act, init=init), 
                Dense(layerwidth, layerwidth, act, init=init), 
                Dense(layerwidth, layerwidth, act, init=init), 
                Dense(layerwidth, 10, identity))
end

########################################################################
# Training function

function train(net, numepochs, batchsize, use_CUDA, flatten_data)
    
    @info "Network contains: $(num_params(net)) trainable params"    

    device = getdevice(use_CUDA)
    net = net |> device
    ps = params(net)
    opt = ADAM(0.0003)
    (trainloader, testloader) = loadMNIST(batchsize, flatten_data)

    for epoch = 1:numepochs
        t1 = time()

        for (x, y) in trainloader
            x, y = x |> device, y |> device
            gs = gradient(()->loss(net(x), y), ps)
            update!(opt, ps, gs)
        end

        train_loss, train_acc = loss_and_accuracy(trainloader, net, device)
        test_loss, test_acc = loss_and_accuracy(testloader, net, device)
    
        runtime = time() - t1
        @info """Epoch: $epoch:    runtime: $(round2(runtime))
        Train:    Acc: $(round2(train_acc*100))%    Loss: $(round6(train_loss))    
        Test:     Acc: $(round2(test_acc*100))%    Loss: $(round6(test_loss))     
    """
    end

end

########################################################################
# Setup experiment

use_CUDA = true
adv_init = true
use_fenbp = true
initmode = adv_init ? glorot_neg_uniform : Flux.glorot_uniform
# my_act = use_fenbp ? my_relu : relu
my_act = use_fenbp ? my_hardtanh : hardtanh

cnnmodel = false
net = cnnmodel ? LeNet5(;act=my_act, init=initmode) : MLP(;act=my_act, layerwidth=512, init=initmode)
flatten_data = cnnmodel ? false : true

numepochs = 20
batchsize= 128
train(net, numepochs, batchsize, use_CUDA, flatten_data)