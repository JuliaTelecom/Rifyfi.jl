
function F1_score_WiSig(Param_Data,Param_Network,Table_Seed_Network,savepathbson="")
    if Param_Network.Train_args.use_cuda ==true 
        hardware1 = "GPU"
    else 
        hardware1 ="CPU"
    end 
    if savepathbson == ""
        if Param_Data.Augmentation_Value.augmentationType == "No_channel"
            savepathbson = "run/WiSig/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)_$(Param_Network.Networkname)/$(Param_Data.txs)_$(Param_Data.rxs)/$(Param_Data.txs)_$(Param_Data.rxs)_$(Param_Data.days)_$(Param_Data.nbSignals)/$(hardware1)"
        else 
            savepathbson = "run/WiSig/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)_$(Param_Network.Networkname)/$(Param_Data.E)_$(Param_Data.S)/$(Param_Data.txs)_$(Param_Data.rxs)/$(Param_Data.days)_$(Param_Data.equalized)_$(Param_Data.nbSignals)_$(Param_Data.nameModel)_$(Param_Data.Augmentation_Value.Channel)_$(Param_Data.Augmentation_Value.Channel_Test)_nbAugment_$(Param_Data.Augmentation_Value.nb_Augment)/$(hardware1)"
        end 
    end 
    if Param_Data.Augmentation_Value.augmentationType == "No_channel"
        savename ="$(Param_Data.txs)_$(Param_Data.rxs)_$(Param_Data.days)_$(Param_Data.equalized)_$(Param_Data.nbSignals)"
    else 
        savename ="$(Param_Data.txs)_$(Param_Data.rxs)_$(Param_Data.days)_$(Param_Data.equalized)_$(Param_Data.nbSignals)_$(Param_Data.name)_$(Param_Data.Augmentation_Value.Channel)_$(Param_Data.Augmentation_Value.Channel_Test)_nbAugment_$(Param_Data.Augmentation_Value.nb_Augment)"
    end 

    dictMarker  = ["square*","triangle*","diamond*","*","pentagon*","rect","otimes","triangle*"];
    # --- Dictionnary for colors 
    dictColor   = ColorSchemes.tableau_superfishel_stone
    @pgf a = Axis({
                height      ="3in",             # Size of Latex object, adapted to IEEE papers 
                width       ="4in",
                grid,
                xlabel      = "Time [s]",       # X axis name 
                ylabel      = "F1 score",       # Y axis name  
                legend_style="{at={(1,0)},anchor=south east,legend cell align=left,align=left,draw=white!15!black}"         # Legend, 2 first parameters are important: we anchor the legend in bottom right (south east) and locate it in bottom right of the figure (1,0)
                },
    );


    mean = zeros(200,5)
    
    for i =1 :1: size(Table_Seed_Network,1)
        Param_Network.Seed_Network= Table_Seed_Network[i]
        name = savename
        Scenario ="$(savepathbson)/F1_Score_$(hardware1)_seed_$(Param_Network.Seed_Network)_dr$(Param_Network.Train_args.dr).csv"
        delim=';'
        nameBase = split(Scenario,".")[1]
        #  ----------------------------------------------------
        # --- Loading the matrix 
        # ----------------------------------------------------- 
        matrice_5 = Matrix(DataFrame(CSV.File(Scenario;delim,types=Float64,header=false)))
        Score_5=zeros(size(matrice_5,1),5)

        Score_5[:,1]=matrice_5[:,1]
        Score_5[:,2]=matrice_5[:,2]
        Score_5[:,3]=matrice_5[:,3]
        Score_5[:,4]=matrice_5[:,4]
        Score_5[:,5]=matrice_5[:,5]
       
        @pgf push!(a,Plot({color=dictColor[i],mark=dictMarker[i]},Table([(Score_5[:,1].-Score_5[1,1]),Score_5[:,2]])))
        @pgf push!(a, LegendEntry("Train")) # Train)
        @pgf push!(a,Plot({color=dictColor[i],mark=dictMarker[i]},Table([(Score_5[:,1].-Score_5[1,1]),Score_5[:,3]])))
        @pgf push!(a, LegendEntry("Test")) # Train)
      
    end
    pgfsave("$(savepathbson)/F1_Score_$(savename).tex",a)

end






function Comput_F1Score_matrix(file)
    m = Matrix(DataFrame(CSV.File(file;delim=';',types=Float64,header=false)))

    # Number of classes 
    nbClass = 6
    # Instantiate metrics 
    tp = zeros(nbClass) # true positive
    fp = zeros(nbClass) # false positive 
    fn = zeros(nbClass) # false negative

        # --- Compute metrics for each class 
    for c = 1 : 1 : nbClass
        # We have c and we guess c 
        tp[c] += m[c,c] 
        # False positive  (equivalent to false alarm in detection theory)
        # We guess c but it was not c 
        fp[c] += sum(m[:,c])-m[c,c]
        # False negative 
        # We predict that it is not c but it was c 
        fn[c] += 100-m[c,c]
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




function Compute_mean(Param_Data,Param_Network,nameSituation,Table_Seed_Network)    
    
    savepath = "run/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)_$(Param_Network.Networkname)/$(Param_Data.E)_$(Param_Data.S)"
    if Param_Network.Train_args.use_cuda ==true 
        hardware1 = "GPU"
    else 
        hardware1 ="CPU"
    end 
    if Param_Data.Augmentation_Value.augmentationType == "No_channel"
        name= "$(Param_Data.E)_$(Param_Data.S)_$(Param_Data.C)_$(Param_Data.RFF)_$(Param_Data.nbSignals)_$(Param_Data.name)"
        else 
        name ="$(Param_Data.E)_$(Param_Data.S)_$(Param_Data.C)_$(Param_Data.RFF)_$(Param_Data.name)_nbAugment_$(Param_Data.Augmentation_Value.nb_Augment)"
        end 
        
    MeanTime =0 
    MeanF1Test = 0
    MeanF1 = 0
    MeanEpoch = 0
    STDTableF1= zeros(size(Table_Seed_Network,1))
    STDTableEpoch= zeros(size(Table_Seed_Network,1))

    for i =1 : 1 : size(Table_Seed_Network,1)
        Param_Network.Seed_Network = Table_Seed_Network[i]
        Scenario ="$(savepath)/$(name)/$(hardware1)/F1_Score_$(hardware1)_seed_$(Param_Network.Seed_Network)_dr$(Param_Network.Train_args.dr).csv"

        delim=';'
        nameBase = split(Scenario,".")[1]
        #  ----------------------------------------------------
        # --- Loading the matrix 
        # ----------------------------------------------------- 
        matrice = Matrix(DataFrame(CSV.File(Scenario;delim,types=Float64,header=false)))
        Score=zeros(size(matrice,1),5)
        Score[:,1]=matrice[:,1] # time 
        Score[:,2]=matrice[:,2] # F1 score Train
        Score[:,3]=matrice[:,3] # F1 score Test
        Value = 500 #size(Score,1)
        MeanTime = Score[Value,1] + MeanTime
        MeanF1 = Score[Value,2] + MeanF1
        MeanF1Test = Score[Value,3] + MeanF1Test
        MeanEpoch = size(Score,1) + MeanEpoch
        STDTableF1[i] = Score[Value,2]
        STDTableEpoch[i] = size(Score,1)
    end 
    MeanTime =  MeanTime / size(Table_Seed_Network,1)
    MeanF1 =  MeanF1 / size(Table_Seed_Network,1)
    MeanF1Test = MeanF1Test / size(Table_Seed_Network,1)
    MeanEpoch = MeanEpoch / size(Table_Seed_Network,1)

    VarF1 = Statistics.std(STDTableF1)
    VarEpoch = Statistics.std(STDTableEpoch)
    @info "Time ", MeanTime
    @info "F1 train ", MeanF1
    @info "F1 test ", MeanF1Test 
    @info "Epoch ", MeanEpoch
    @info "Variance F1 ",  VarF1
    @info " Variance epoch ", VarEpoch
    
end 



function Confusion_Matrix_CSV_WiSig(Param_Data,Param_Network,Param_Data_test,savepathbson="")
    
    if Param_Network.Train_args.use_cuda ==true 
        hardware1 = "GPU"
    else 
        hardware1 ="CPU"
    end 
    if savepathbson == ""
        if Param_Data.Augmentation_Value.augmentationType == "No_channel"
            savepathbson = "run/WiSig/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)_$(Param_Network.Networkname)/$(Param_Data.txs)_$(Param_Data.rxs)/$(Param_Data.txs)_$(Param_Data.rxs)_$(Param_Data.days)_$(Param_Data.equalized)_$(Param_Data.nbSignals)/$(hardware1)"
        else 
            savepathbson = "run/WiSig/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)_$(Param_Network.Networkname)/$(Param_Data.txs)_$(Param_Data.rxs)/$(Param_Data.txs)_$(Param_Data.rxs)_$(Param_Data.days)_$(Param_Data.equalized)_$(Param_Data.nbSignals)_$(Param_Data.Augmentation_Value.Channel)_$(Param_Data.Augmentation_Value.Channel_Test)_nbAugment_$(Param_Data.Augmentation_Value.nb_Augment)/$(hardware1)"
        end 
    end 

    allAcc = Float64[]
    res = RiFyFi_IdF.loadCNN("$(savepathbson)/model_seed_$(Param_Network.Seed_Network)_dr$(Param_Network.Train_args.dr).bson")

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
