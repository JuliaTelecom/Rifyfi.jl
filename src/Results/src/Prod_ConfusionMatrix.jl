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
    filename = .pkl
     file 
    txs,rxs,days,equalized = param for test with 2 for equalized
    rxsnn,daysnn = param to choose to CNN 
    ChunSize = 256 
    dataAug = "sans" or "OfflineV1" 
"""

function Confusion_Matrix_CSV(Param_Data,Param_Network,Param_Data_test,savepathbson="")
    if Param_Network.Train_args.use_cuda ==true 
        hardware1 = "GPU"
    else 
        hardware1 ="CPU"
    end 
    if savepathbson == ""
        if Param_Data.Augmentation_Value.augmentationType == "No_channel"
            savepath_model = "run/Synth/$(Param_Data.Modulation)/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)_$(Param_Network.Networkname)/$(Param_Data.E)_$(Param_Data.S)/$(Param_Data.E)_$(Param_Data.S)_$(Param_Data.C)_$(Param_Data.RFF)_$(Param_Data.nbSignals)_$(Param_Data.nameModel)/$(hardware1)"
        else 
            savepath_model = "run/Synth/$(Param_Data.Modulation)/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)_$(Param_Network.Networkname)/$(Param_Data.E)_$(Param_Data.S)/$(Param_Data.E)_$(Param_Data.S)_$(Param_Data.C)_$(Param_Data.RFF)_$(Param_Data.nbSignals)_$(Param_Data.nameModel)_$(Param_Data.Augmentation_Value.Channel)_$(Param_Data.Augmentation_Value.Channel_Test)_nbAugment_$(Param_Data.Augmentation_Value.nb_Augment)/$(hardware1)"
        end 
    end 

    allAcc = Float64[]
        
    res = RiFyFi_IdF.loadCNN("$(savepath_model)/model_seed_$(Param_Network.Seed_Network)_dr$(Param_Network.Train_args.dr)_$(Param_Data.Modulation).bson")

    model = res.model
    testmode!(model, true)  # We are in test mode, with no dropout 
    (moy,std_val) = (nothing,nothing)
    allAccuracy = Float64[]
    (_,_,X_test,Y_test) = RiFyFi_VDG.loadCSV_Synthetic(Param_Data_test) 

    
    if Param_Network.Train_args.use_cuda
        device= gpu
    else
        device =cpu
    end
    dataTest  = Flux.Data.DataLoader((X_test, Y_test), batchsize = Param_Network.Train_args.batchsize, shuffle = true )
    l̂,l = inference(model,dataTest,device)
    acc = getAccuracy(l̂,l) 
    @info acc
    


    confMatrix = confusionMatrix(l̂,l,Param_Data.nbTx)
    plt = plotConfusionMatrix(confMatrix )

    savepath_result ="Results/$(Param_Data.Modulation)/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)_$(Param_Network.Networkname)/$(Param_Data.E)_$(Param_Data.S)"
    !ispath(savepath_result) && mkpath(savepath_result)
        Temp=zeros(1,Param_Data.nbTx)
        if Param_Data.Augmentation_Value.augmentationType == "No_channel"
            file="$(savepath_result)/confMatrix_$(Param_Data.E)_$(Param_Data.S)_$(Param_Data.C)_$(Param_Data.RFF)_$(Param_Data.nbSignals)_$(Param_Data.name)_seed_$(Param_Network.Seed_Network).csv"
        else 
            file="$(savepath_result)/confMatrix_$(Param_Data.E)_$(Param_Data.S)_$(Param_Data.C)_$(Param_Data.RFF)_$(Param_Data.nbSignals)_$(Param_Data.name)_$(Param_Data.Augmentation_Value.Channel)_$(Param_Data.Augmentation_Value.Channel_Test)_nbAugment_$(Param_Data.Augmentation_Value.nb_Augment)_seed_$(Param_Network.Seed_Network).csv"
        end 
        open(file,"w") do io
            for i in 0:size(confMatrix,1)-1
                Temp[1,:]  = round.(confMatrix[i+1,:]*100;digits=1)
                writedlm(io,[vcat((Temp))],';')  #Ecriture Re-Im
            end 
        end
     MainPlottingMatrix_Latex(file,Param_Data.E,Param_Data.S,Param_Data.C,Param_Network.Networkname,Param_Data.RFF,Param_Data.Chunksize,Param_Network.Train_args.batchsize,Param_Data.nbTx)


    ################################################
    # ------ CFO dynamic --------
    ################################################

    if Param_Data.RFF == "all_impairments_dynamic_cfo"
        (_,_,X_test_dyn,Y_test_dyn) = RiFyFi_VDG.loadCSV_Synthetic_dynCFO(Param_Data_test) 
        if Param_Network.Train_args.use_cuda
            device= gpu
        else
            device =cpu
        end
        dataTest_dyn  = Flux.Data.DataLoader((X_test_dyn, Y_test_dyn), batchsize = Param_Network.Train_args.batchsize, shuffle = true )
        l̂,l = inference(model,dataTest_dyn,device)
        acc = getAccuracy(l̂,l) 
        
        @info acc
        confMatrix = confusionMatrix(l̂,l,Param_Data.nbTx)
        plt = plotConfusionMatrix(confMatrix )

        savepath_result = "Results/$(Param_Data.Modulation)/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)_$(Param_Network.Networkname)/$(Param_Data.E)_$(Param_Data.S)"

        !ispath(savepath_result) && mkpath(savepath_result)
            Temp=zeros(1,Param_Data.nbTx)
            if Param_Data.Augmentation_Value.augmentationType == "No_channel"
                file="$(savepath_result)/confMatrix_$(Param_Data.E)_$(Param_Data.S)_$(Param_Data.C)_$(Param_Data.RFF)_$(Param_Data.nbSignals)_$(Param_Data.name)_seed_$(Param_Network.Seed_Network)_dyn.csv"
            else 
                file="$(savepath_result)/confMatrix_$(Param_Data.E)_$(Param_Data.S)_$(Param_Data.C)_$(Param_Data.RFF)_$(Param_Data.nbSignals)_$(Param_Data.name)_$(Param_Data.Augmentation_Value.Channel)_$(Param_Data.Augmentation_Value.Channel_Test)_nbAugment_$(Param_Data.Augmentation_Value.nb_Augment)_seed_$(Param_Network.Seed_Network)_dyn.csv"
            end 
            open(file,"w") do io
                for i in 0:size(confMatrix,1)-1
                    Temp[1,:]  = round.(confMatrix[i+1,:]*100;digits=1)
                    writedlm(io,[vcat((Temp))],';')  #Ecriture Re-Im
                end 
            end
        MainPlottingMatrix_Latex(file,Param_Data.E,Param_Data.S,Param_Data.C,Param_Network.Networkname,Param_Data.RFF,Param_Data.Chunksize,Param_Network.Train_args.batchsize,Param_Data.nbTx)
    end 
end




function MainPlottingMatrix_Latex(file,E="E2",S="S1",C="C1",Network="AlexNet",RFF="all_impairments",ChunkSize=256,batchsize=100,nbRadioTx=4)
    # --- Create the header
    header      = (["Tx\$_$i\$" for i = 1 : 1 : nbRadioTx])
    row_labels  = (["TxTrue\$_$i\$" for i = 1 : 1 : nbRadioTx])
    row_label_column_title = "\\backslashbox[15mm][]{\\footnotesize True}{\\footnotesize Guess}"
        cellcolor_header = "blue!10!white"      

        m = latexConfusionMatrix(file;doSave=true,standalone=true,header,row_label_column_title,row_labels,cellcolor_header)
end 