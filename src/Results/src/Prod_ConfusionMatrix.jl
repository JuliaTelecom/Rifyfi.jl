# ----------------------------------------------------
# --- Package managment
# ---------------------------------------------------- 
# --- Import classic packages




""" 
Compute confusion matrix when inputs are the estimated labels and the true labels 
"""
function confusionMatrix(l̂::AbstractArray,l::AbstractArray,nbRadios::Number)
    confMatrix = zeros(nbRadios,nbRadios)
    @assert size(l̂) == size(l) "Estimated radio label and true radio label must have the same size: here we have $(length(l̂)) and $(length(l))"
    for n ∈ eachindex(l) 
        confMatrix[l[n],l̂[n]] += 1 
    end
    # --- Normalize to have probability 
    for n in 1 : 1 : nbRadios
        confMatrix[n,:] = confMatrix[n,:] ./ sum(confMatrix[n,:])
    end
    
    return confMatrix 
end



"""
Returns accuracy estimation (in percent), based on onecold estimator (label vector)
"""
getAccuracy(a,b) = sum( a .== b) / length(b)*100


function plotConfusionMatrix(confMatrix)
    # Define a colormap to be sure low proba and high proba are visible 
    # col= cgrad(:thermal,[40/100 99.5/100])
    col= cgrad(:thermal)
    # Plotting heatmap 
    plt= heatmap(confMatrix,c=col,size=(1200,900))
    # Adding labels 
    fontsize = 15
    nrow, ncol = size(confMatrix)
    ann = [(i,j, text(round(confMatrix[j,i]*100, digits=2), fontsize, :white, :center))
           for i in 1:nrow for j in 1:ncol] # Switch i and j as matrix is printed up (otherwise we invert color and labels)
                annotate!(ann, linecolor=:white)
                ann = [(i,i, text(round(confMatrix[i,i]*100, digits=2), fontsize, :black, :center))
           for i in 1:nrow ]
    annotate!(ann, linecolor=:white)
    xlabel!("Infered Radio index")
    ylabel!("Labeled Radio index")
    return plt 
end


""" Fonction qui plot la matrice de confusion en format csv \n 
    filename = .pkl file 
    txs,rxs,days,equalized = param for test with 2 for equalized
    rxsnn,daysnn = param to choose to CNN 
    ChunSize = 256 
    dataAug = "No_channel" or "OfflineV1" 
"""

function Confusion_Matrix_CSV(Param_Data,Param_Network,Param_Data_test,savepathbson="")
    if Param_Network.Train_args.use_cuda ==true 
        hardware1 = "GPU"
    else 
        hardware1 ="CPU"
    end 
    if savepathbson == ""
        if Param_Data.Augmentation_Value.augmentationType == "No_channel"
            savepathbson = "run/Synth/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)_$(Param_Network.Networkname)/$(Param_Data.E)_$(Param_Data.S)/$(Param_Data.E)_$(Param_Data.S)_$(Param_Data.C)_$(Param_Data.RFF)_$(Param_Data.nbSignals)_$(Param_Data.nameModel)/$(hardware1)"
        else 
            savepathbson = "run/Synth/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)_$(Param_Network.Networkname)/$(Param_Data.E)_$(Param_Data.S)/$(Param_Data.E)_$(Param_Data.S)_$(Param_Data.C)_$(Param_Data.RFF)_$(Param_Data.nbSignals)_$(Param_Data.nameModel)_$(Param_Data.Augmentation_Value.Channel)_$(Param_Data.Augmentation_Value.Channel_Test)_nbAugment_$(Param_Data.Augmentation_Value.nb_Augment)/$(hardware1)"
        end 
    end 

    allAcc = Float64[]
        
    res =RiFyFi_IdF.loadCNN("$(savepathbson)/model_seed_$(Param_Network.Seed_Network)_dr$(Param_Network.Train_args.dr).bson")

    model = res.model
    testmode!(model, true)  # We are in test mode, with no dropout 
    (moy,std_val) = (nothing,nothing)
    allAccuracy = Float64[]
   
    (_,_,X_test,Y_test) =loadCSV_Synthetic(Param_Data_test) 

    
    if Param_Network.Train_args.use_cuda
        device= gpu
    else
        device =cpu
    end
    dataTest  = Flux.Data.DataLoader((X_test, Y_test), batchsize = Param_Network.Train_args.batchsize, shuffle = true )
    l̂,l = inference(model,dataTest,device)
    #acc = getAccuracy(l̂,l) 
    #@info acc
    confMatrix = confusionMatrix(l̂,l,Param_Data.nbTx)
    plt = plotConfusionMatrix(confMatrix )

    savepath ="Results/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)_$(Param_Network.Networkname)/$(Param_Data.E)_$(Param_Data.S)"
    !ispath(savepath) && mkpath(savepath)
        Temp=zeros(1,Param_Data.nbTx)
        if Param_Data.Augmentation_Value.augmentationType == "No_channel"
            file="$(savepath)/confMatrix_$(Param_Data.E)_$(Param_Data.S)_$(Param_Data.C)_$(Param_Data.RFF)_$(Param_Data.nbSignals)_$(Param_Data.name)_seed_$(Param_Network.Seed_Network).csv"
        else 
            file="$(savepath)/confMatrix_$(Param_Data.E)_$(Param_Data.S)_$(Param_Data.C)_$(Param_Data.RFF)_$(Param_Data.nbSignals)_$(Param_Data.name)_$(Param_Data.Augmentation_Value.Channel)_$(Param_Data.Augmentation_Value.Channel_Test)_nbAugment_$(Param_Data.Augmentation_Value.nb_Augment)_seed_$(Param_Network.Seed_Network).csv"
        end 
        open(file,"w") do io
            for i in 0:size(confMatrix,1)-1
                Temp[1,:]  = round.(confMatrix[i+1,:]*100;digits=1)
                writedlm(io,[vcat((Temp))],';')  #Ecriture Re-Im
            end 
        end
     MainPlottingMatrix_Latex(file,Param_Data.nbTx,Param_Data.E,Param_Data.S,Param_Data.C,Param_Network.Networkname,Param_Data.RFF,Param_Data.Chunksize,Param_Network.Train_args.batchsize)
end


function Confusion_Matrix_CSV_WiSig(Param_Data,Param_Network,Param_Data_test,savepathbson="")
    
    if Param_Network.Train_args.use_cuda ==true 
        hardware1 = "GPU"
    else 
        hardware1 ="CPU"
    end 
    if savepathbson == ""
       # if Param_Data.Augmentation_Value.augmentationType == "No_channel"
            savepathbson = "run/WiSig/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)_$(Param_Network.Networkname)/$(Param_Data.txs)_$(Param_Data.rxs)/$(Param_Data.txs)_$(Param_Data.rxs)_$(Param_Data.days)_$(Param_Data.equalized)_$(Param_Data.nbSignals)/$(hardware1)"
       # else 
       #     savepathbson = "run/WiSig/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)_$(Param_Network.Networkname)/$(Param_Data.txs)_$(Param_Data.rxs)/$(Param_Data.txs)_$(Param_Data.rxs)_$(Param_Data.days)_$(Param_Data.nbSignals)_$(Param_Data.Augmentation_Value.Channel)_$(Param_Data.Augmentation_Value.Channel_Test)_nbAugment_$(Param_Data.Augmentation_Value.nb_Augment)/$(hardware1)"
       # end 
    end 

    allAcc = Float64[]
        
    res =RiFyFi_IdF.loadCNN("$(savepathbson)/model_seed_$(Param_Network.Seed_Network)_dr$(Param_Network.Train_args.dr).bson")

    model = res.model
    testmode!(model, true)  # We are in test mode, with no dropout 
    (moy,std_val) = (nothing,nothing)
    allAccuracy = Float64[]
   
    (_,_,X_test,Y_test) =WiSig_Database.loadCSV_WiSig(Param_Data_test) 

    
    if Param_Network.Train_args.use_cuda
        device= gpu
    else
        device =cpu
    end
    device =cpu
    dataTest  = Flux.Data.DataLoader((X_test, Y_test), batchsize = Param_Network.Train_args.batchsize, shuffle = true )
    
    l̂,l = inference(model,dataTest,device)
    acc = getAccuracy(l̂,l) 
    @info "acc" acc
    confMatrix = confusionMatrix(l̂,l,Param_Data.nbTx)
    plt = plotConfusionMatrix(confMatrix )

    savepath ="Results/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)_$(Param_Network.Networkname)/$(Param_Data.txs)_$(Param_Data.rxs)"
    !ispath(savepath) && mkpath(savepath)
        Temp=zeros(1,Param_Data.nbTx)
      #  if Param_Data.Augmentation_Value.augmentationType == "No_channel"
            file="$(savepath)/confMatrix_$(Param_Data.txs)_$(Param_Data.rxs)_$(Param_Data.days)_$(Param_Data.equalized)_$(Param_Data.nbSignals)_$(Param_Data.name)_seed_$(Param_Network.Seed_Network)_$(Param_Data_test.days).csv"
        #else 
        #    file="$(savepath)/confMatrix_$(Param_Data.E)_$(Param_Data.S)_$(Param_Data.C)_$(Param_Data.RFF)_$(Param_Data.nbSignals)_$(Param_Data.name)_$(Param_Data.Augmentation_Value.Channel)_$(Param_Data.Augmentation_Value.Channel_Test)_nbAugment_$(Param_Data.Augmentation_Value.nb_Augment)_seed_$(Param_Network.Seed_Network).csv"
        #end 
        open(file,"w") do io
            for i in 0:size(confMatrix,1)-1
                Temp[1,:]  = round.(confMatrix[i+1,:]*100;digits=1)
                writedlm(io,[vcat((Temp))],';')  #Ecriture Re-Im
            end 
        end
     MainPlottingMatrix_Latex(file,Param_Data.nbTx)
     return acc
end


function Confusion_Matrix_CSV_Oracle(Param_Data,Param_Network,Param_Data_test,savepathbson="")
    
    if Param_Network.Train_args.use_cuda ==true 
        hardware1 = "GPU"
    else 
        hardware1 ="CPU"
    end 
    if savepathbson == ""
        if Param_Data.Augmentation_Value.augmentationType == "No_channel"
            savepathbson = "run/Oracle/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)_$(Param_Network.Networkname)/$(Param_Data.nbSignals)_$(Param_Data.distance)/$(hardware1)"
        else 
            savepathbson = "run/Oracle/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)_$(Param_Network.Networkname)/$(Param_Data.nbSignals)_$(Param_Data.Augmentation_Value.Channel)_$(Param_Data.Augmentation_Value.Channel_Test)_nbAugment_$(Param_Data.Augmentation_Value.nb_Augment)/$(hardware1)"
        end 
    end 

    allAcc = Float64[]
        
    res =RiFyFi_IdF.loadCNN("$(savepathbson)/model_seed_$(Param_Network.Seed_Network)_dr$(Param_Network.Train_args.dr).bson")

    model = res.model
    testmode!(model, true)  # We are in test mode, with no dropout 
    (moy,std_val) = (nothing,nothing)
    allAccuracy = Float64[]
   
    (_,_,X_test,Y_test) =Oracle_Database.loadCSV_Oracle(Param_Data_test) 

    
    if Param_Network.Train_args.use_cuda
        device= gpu
    else
        device =cpu
    end
    dataTest  = Flux.Data.DataLoader((X_test, Y_test), batchsize = Param_Network.Train_args.batchsize, shuffle = true )
    l̂,l = inference(model,dataTest,device)
    acc = getAccuracy(l̂,l) 
    @info "acc" acc
    confMatrix = confusionMatrix(l̂,l,Param_Data.nbTx)
    plt = plotConfusionMatrix(confMatrix )

    savepath ="Results/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)_$(Param_Network.Networkname)/"
    !ispath(savepath) && mkpath(savepath)
        Temp=zeros(1,Param_Data.nbTx)
        if Param_Data.Augmentation_Value.augmentationType == "No_channel"
            file="$(savepath)/confMatrix_$(Param_Data.nbSignals)_$(Param_Data.name)_seed_$(Param_Network.Seed_Network).csv"
        else 
            file="$(savepath)/confMatrix_$(Param_Data.nbSignals)_$(Param_Data.name)_$(Param_Data.Augmentation_Value.Channel)_$(Param_Data.Augmentation_Value.Channel_Test)_nbAugment_$(Param_Data.Augmentation_Value.nb_Augment)_seed_$(Param_Network.Seed_Network).csv"
        end 
        open(file,"w") do io
            for i in 0:size(confMatrix,1)-1
                Temp[1,:]  = round.(confMatrix[i+1,:]*100;digits=1)
                writedlm(io,[vcat((Temp))],';')  #Ecriture Re-Im
            end 
        end
     MainPlottingMatrix_Latex(file,Param_Data.nbTx)
end


function Confusion_Matrix_CSV_Exp(Param_Data,Param_Network,Param_Data_test,savepathbson="")
    
    if Param_Network.Train_args.use_cuda ==true 
        hardware1 = "GPU"
    else 
        hardware1 ="CPU"
    end 
    if savepathbson == ""
        if Param_Data.Augmentation_Value.augmentationType == "No_channel"
            if Param_Data.permutation == true
                savepathbson = "run/Experiment/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)_$(Param_Network.Networkname)_$(Param_Data.Type_of_sig)/Run$(Param_Data.run)_Test$(Param_Data.Test)_$(Param_Data.nbTx)_$(Param_Data.nbSignals)_permut/$(hardware1)"
            else 
                savepathbson = "run/Experiment/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)_$(Param_Network.Networkname)_$(Param_Data.Type_of_sig)/Run$(Param_Data.run)_Test$(Param_Data.Test)_$(Param_Data.nbTx)_$(Param_Data.nbSignals)/$(hardware1)"
            end        
        else 
            savepathbson = "run/Experiment/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)_$(Param_Network.Networkname)/$(Param_Data.nbSignals)_$(Param_Data.Augmentation_Value.Channel)_$(Param_Data.Augmentation_Value.Channel_Test)_nbAugment_$(Param_Data.Augmentation_Value.nb_Augment)/$(hardware1)"
        end 
    end 


  
    allAcc = Float64[]
        
    res =RiFyFi_IdF.loadCNN("$(savepathbson)/model_seed_$(Param_Network.Seed_Network)_dr$(Param_Network.Train_args.dr).bson")

    model = res.model
    testmode!(model, true)  # We are in test mode, with no dropout 
    (moy,std_val) = (nothing,nothing)
    allAccuracy = Float64[]
   
    (_,_,X_test,Y_test) =Experiment_Database.loadCSV_Exp(Param_Data_test) 

    
    if Param_Network.Train_args.use_cuda
        device= gpu
    else
        device =cpu
    end
    dataTest  = Flux.Data.DataLoader((X_test, Y_test), batchsize = Param_Network.Train_args.batchsize, shuffle = true )
    l̂,l = inference(model,dataTest,device)
    acc = getAccuracy(l̂,l) 
    @info "acc" acc
    confMatrix = confusionMatrix(l̂,l,Param_Data.nbTx)
    plt = plotConfusionMatrix(confMatrix )

    if Param_Data.permutation==true 
        savepath ="Results/Exp/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)_$(Param_Network.Networkname)_$(Param_Data.Type_of_sig)/RunTrain$(Param_Data.run)_RunTest$(Param_Data_test.run)_Test$(Param_Data_test.Test)_permut/"
    else 
        savepath ="Results/Exp/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)_$(Param_Network.Networkname)_$(Param_Data.Type_of_sig)/RunTrain$(Param_Data.run)_RunTest$(Param_Data_test.run)_Test$(Param_Data_test.Test)/"
    end 
    !ispath(savepath) && mkpath(savepath)
        Temp=zeros(1,Param_Data.nbTx)
        if Param_Data.Augmentation_Value.augmentationType == "No_channel"
            file="$(savepath)/confMatrix_$(Param_Data.nbSignals)_$(Param_Data.name)_seed_$(Param_Network.Seed_Network).csv"
        else 
            file="$(savepath)/confMatrix_$(Param_Data.nbSignals)_$(Param_Data.name)_$(Param_Data.Augmentation_Value.Channel)_$(Param_Data.Augmentation_Value.Channel_Test)_nbAugment_$(Param_Data.Augmentation_Value.nb_Augment)_seed_$(Param_Network.Seed_Network).csv"
        end 
        open(file,"w") do io
            for i in 0:size(confMatrix,1)-1
                Temp[1,:]  = round.(confMatrix[i+1,:]*100;digits=1)
                writedlm(io,[vcat((Temp))],';')  #Ecriture Re-Im
            end 
        end
     MainPlottingMatrix_Latex(file,Param_Data.nbTx)
end

function MainPlottingMatrix_Latex(file,nbRadioTx=4,E="E2",S="S1",C="C1",Network="AlexNet",RFF="all_impairments",ChunkSize=256,batchsize=100)
    # --- Create the header
    header      = (["Tx\$_$i\$" for i = 1 : 1 : nbRadioTx])
    row_labels  = (["TxTrue\$_$i\$" for i = 1 : 1 : nbRadioTx])
    row_label_column_title = "\\backslashbox[15mm][]{\\footnotesize True}{\\footnotesize Guess}"
       # file="run/ManySig/sans_$(nbRadioTx)_$(ChunkSize)_$(Network)/$(hardware)/confMatrix_$(E)_$(S)_$(C)_$(C_test)_$(RFF).csv"
        cellcolor_header = "blue!10!white"      

        m = latexConfusionMatrix(file;doSave=true,standalone=true,header,row_label_column_title,row_labels,cellcolor_header)
end 