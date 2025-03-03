
function F1_score_Oracle(Param_Data,Param_Network,Table_Seed_Network,savepathbson="")
    
    if Param_Network.Train_args.use_cuda ==true 
        hardware1 = "GPU"
    else 
        hardware1 ="CPU"
    end 

    savepathbson = "run/Oracle/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)_$(Param_Network.Networkname)/$(Param_Data.nbSignals)_$(Param_Data.distance)/$(hardware1)"

    savename ="F1_Score_$(hardware1)_seed_$(Param_Network.Seed_Network)_dr$(Param_Network.Train_args.dr)"

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
        @pgf push!(a, LegendEntry("Seed $(i)")) # Train)
      
    end
    pgfsave("$(savepathbson)/F1_Score_$(savename).tex",a)

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

