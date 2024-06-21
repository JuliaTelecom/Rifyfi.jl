""" 
Returns the loss function and the accuracy of a loader (data) in a model for a given loss function 
The model can be done on CPU or GPU
"""
function eval_loss_accuracy(loader, model, loss, device)
    l = 0f0
    acc = 0
    ntot = 0
    for (x, y) in loader
        x, y = x |> device, y |> device
      #  ŷ = model(reshape(x, (256,2,1,(size(x))[3])))    
        ŷ = model(x)
        l += loss(ŷ, y) * size(x)[end]        
        acc += sum(onecold(ŷ |> cpu) .== onecold(y |> cpu))
        ntot += size(x)[end]
    end
    return (loss = l/ntot |> round4, acc = acc/ntot*100 |> round4)
end


""" Returns the macro average F1 score for the data contained in the dataloader using the learning model `model`. Use the device `device`to speed up is possible (e.g device=gpu)
Returns a F1 score with macro-average calculation. Recall and precision are computed per class and the results is then average. it corresponds to have same weigth for all classes whatever the number of occurence per class there is.
"""
function F1Score(loader,model,device=cpu)
    # Number of classes 
    nbClass = size(loader.data[2],1)
    # Instantiate metrics 
    tp = zeros(nbClass) # true positive
    fp = zeros(nbClass) # false positive 
    fn = zeros(nbClass) # false negative
    for (x, y) in loader
        # Load to CPU
        xd = x |> device
        # Train, bring back and decision
      #  ŷ = model(reshape(xd, (256,2,1,(size(xd))[3]))) |> cpu  |> onecold
      
        ŷ = model(xd) |> cpu  |> onecold
        y = onecold(y)
        # --- Compute metrics for each class 
        for c = 1 : 1 : nbClass
            cTrue = findall(y .== c)  # Corresponds to true radio c 
            cGuess = findall(ŷ .== c) # Guessed as radio c 
            # True positive 
            # We have c and we guess c 
            tp[c] += sum(ŷ[cTrue] .== c)
            # False positive  (equivalent to false alarm in detection theory)
            # We guess c but it was not c 
            fp[c] += sum(y[cGuess] .!= c)
            # False negative 
            # We predict that it is not c but it was c 
            fn[c] += sum(ŷ[cTrue] .!= c)
        end 
    end
    # Macro average approach, calculate precision and recall per class  and average
    # Mirror to what is done in Gegelati, with same argument 
    # (chosen instead of the global f1 score as it gives an equal weight to
    # the f1 score of each class, no matter its ratio within the observed
    # population)
    replace_nan!(x) = isnan(x) ? 0 : x
    precision = (tp ./ (tp .+ fp)) 
    recall    = (tp ./ (tp .+ fn))
    F1        = replace_nan!.(2 ./ (1 ./precision + 1 ./recall))|> mean
    
    return F1
end






function customTrain!(dataTrain,dataTest,savepath_model,Param_Network,Modulation,dataTrain_dyn=0,dataTest_dyn= 0)
    # ----------------------------------------------------
    # --- CPU or GPU
    # ---------------------------------------------------- 
    use_cuda = Param_Network.Train_args.use_cuda && CUDA.functional()
    if use_cuda
        device = gpu
        DeviceName = "GPU"
        @info "Training on GPU"
    else
        device = cpu
        DeviceName = "CPU"
        @info "Training on CPU"
    end

    # --- Loading model
    model = Param_Network.model |> device
    
    @info "Chosen NN model: $(num_params(model)) trainable params"    
    # --- Optimizer
    ps = Flux.params(model)  
    opt = ADAM(Param_Network.Train_args.η) 

    # ----------------------------------------------------
    # --- Closure for performance evaluation
    # --- Comment if you are on specific hardware 
    # ---------------------------------------------------- 
    function report(epoch)
        train = eval_loss_accuracy(dataTrain, model, Param_Network.loss, device)
        test = eval_loss_accuracy(dataTest, model,Param_Network.loss, device)        
        println("Epoch: $epoch   Train: $(train)   Test: $(test)")
        return (train.loss,train.acc,test.loss,test.acc)
    end
    
    # ----------------------------------------------------
    # --- Training
    # ---------------------------------------------------- 
    @info "Start Training"
    ###### Comment if need ###################################################
    report(0) 
    trainLoss = Float32[]
    trainAcc= Float32[]
    testLoss  = Float32[]
    testAcc   = Float32[]
    ################################################################################
    trainf1 = Float32[]
    testf1 = Float32[] 
    testf1_dyn = Float32[] 
    ta=0
    epoch =0
    Random.seed!(Param_Network.Seed_Network)
    while epoch < Param_Network.Train_args.epochs && ta < 98.0
        epoch = epoch+1
        Param_Network.Train_args.tInit = time() # Set up time of origin
        # Train 
        @showprogress for (x, y) in dataTrain   
            x, y = x |> device, y |> device        
            gs = Flux.gradient(ps) do
                ŷ = model(x)
                Param_Network.loss(ŷ, y)
            end
            Flux.Optimise.update!(opt, ps, gs)
        end
        Param_Network.Train_args.timings[epoch] = round(time() - Param_Network.Train_args.tInit;digits=6) # mus resolution
        _f1 = F1Score(dataTrain,model,device)
        @info _f1  
        push!(trainf1,_f1)
        _f1 = F1Score(dataTest,model,device)
        @info _f1  
        push!(testf1,_f1)
        if dataTest_dyn != 0
            @info "hee"
            _f1 = F1Score(dataTest_dyn,model,device)
            @info _f1  
            push!(testf1_dyn,_f1)
        else 
            push!(testf1_dyn,0)
        end 
        # A Commenter si besoin #########################################################
        if epoch % Param_Network.Train_args.infotime == 0
            (tl,ta,el,ea) = report(epoch)
            push!(trainLoss,tl)
            push!(trainAcc,ta)
            push!(testLoss,el)
            push!(testAcc,ea)
            end  
        ################################################################################
    end
    # Timings are calculated per eopch, we want complete time 
    Param_Network.Train_args.timings = cumsum(Param_Network.Train_args.timings;dims=1)
    # Write timings and accuracy in a file 
    open("$(savepath_model)/F1_Score_$(DeviceName)_seed_$(Param_Network.Seed_Network)_dr$(Param_Network.Train_args.dr)_$(Modulation).csv","w") do io 
        arr = [Param_Network.Train_args.timings[1:epoch] trainf1 testf1 testf1_dyn trainLoss testLoss]
        writedlm(io,round.(arr;digits=4),';')
    end
    # --- Copy NN model to CPU 
    nn = model|> cpu
    # --- Return 
    return nn,trainLoss,trainAcc,testLoss,testAcc,Param_Network.Train_args
end




""" 
Returns the number of parameters to be estimated in a Flux network 
"""
num_params(model) = sum(length, Flux.params(model)) 


""" 
Round a number around 4 digits 
""" 
round4(x) = round(x, digits=4)
