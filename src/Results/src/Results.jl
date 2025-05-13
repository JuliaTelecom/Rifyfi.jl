module Results
""" Create Results: 
- F1 score function of time
- Confusion Matrix
- Compute the mean value 
"""

using BSON 
using Plots 
using PrettyTables         
using DelimitedFiles
using PGFPlotsX 
using Colors, ColorSchemes 
using Random
using Flux
using CSV
using DataFrames
using Infiltrator
gr()  
include("../../LatexConfusionMatrix/src/LatexConfusionMatrix.jl")
using .LatexConfusionMatrix

include("Prod_ConfusionMatrix.jl")
include("VDG_Results.jl")
include("Oracle_Results.jl")
include("WiSig_Results.jl")
include("Exp_Results.jl")



include("../../Augmentation/src/Augmentation.jl")
using .Augmentation

include("../../RiFyFi_VDG/src/RiFyFi_VDG.jl")
using .RiFyFi_VDG

include("../../RiFyFi_IdF/src/RiFyFi_IdF.jl")
using .RiFyFi_IdF

include("../../WiSig_Database/src/WiSig_Database.jl")
using .WiSig_Database

include("../../Oracle_Database/src/Oracle_Database.jl")
using .Oracle_Database
include("../../Experiment_Database/src/Experiment_Database.jl")
using .Experiment_Database

function main(Param_Data,Param_Network,Type_Resuts,savepathbson,Param_Data_test,Table_Seed_Network)
    acc= 0
    # WiSig 
    if Param_Data.name == "WiSig"
        if Type_Resuts == "F1_score"
            F1_score_WiSig(Param_Data,Param_Network,Table_Seed_Network,savepathbson)
        elseif Type_Resuts == "Compute_mean"
            Compute_mean_WiSig(Param_Data,Param_Network,nameSituation,Table_Seed_Network)
        elseif Type_Resuts == "Confusion_Matrix"
            acc = Confusion_Matrix_CSV_WiSig(Param_Data,Param_Network,Param_Data_test,savepathbson )
        end 
    # Oracle 
    elseif Param_Data.name == "Oracle"
        if Type_Resuts == "Confusion_Matrix"
            Confusion_Matrix_CSV_Oracle(Param_Data,Param_Network,Param_Data_test,savepathbson )
        end 
    # Experiment
    elseif Param_Data.name == "Exp"
        if Type_Resuts == "Confusion_Matrix"
            acc= Confusion_Matrix_CSV_Exp(Param_Data,Param_Network,Param_Data_test,savepathbson )
        elseif Type_Resuts == "time"
            Time_Exp(Param_Data,Param_Network,Param_Data_test,savepathbson )
        end 
    else 
        if Type_Resuts == "F1_score"
            F1_score_Synth(Param_Data,Param_Network,Table_Seed_Network,savepathbson)
        elseif Type_Resuts == "Compute_mean"
            Compute_mean(Param_Data,Param_Network,nameSituation,Table_Seed_Network)
        elseif Type_Resuts == "Confusion_Matrix"
            Confusion_Matrix_CSV(Param_Data,Param_Network,Param_Data_test,savepathbson )
        
        end 
    end 
    return acc
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
            elseif Param_Data.noise==nothing 
                savepathbson = "run/Experiment/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)_$(Param_Network.Networkname)_$(Param_Data.Type_of_sig)/Run$(Param_Data.run)_Test$(Param_Data.Test)_$(Param_Data.nbTx)_$(Param_Data.nbSignals)/$(hardware1)"
            else 
                savepathbson = "run/Experiment/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)_$(Param_Network.Networkname)_$(Param_Data.Type_of_sig)/Run$(Param_Data.run)_Test$(Param_Data.Test)_$(Param_Data.nbTx)_$(Param_Data.nbSignals)_$(Param_Data.noise)/$(hardware1)"

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
        savepath ="Results/Exp/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)_$(Param_Network.Networkname)_$(Param_Data.Type_of_sig)/RunTrain$(Param_Data.run)_RunTest$(Param_Data_test.run)_Test$(Param_Data_test.Test)_$(Param_Data_test.noise)/"
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

   return acc
end


#=
function F1_score_Synth(Param_Data,Param_Network,Table_Seed_Network,savepathbson="")
    
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
    if Param_Data.Augmentation_Value.augmentationType == "No_channel"
        savename ="$(Param_Data.E)_$(Param_Data.S)_$(Param_Data.C)_$(Param_Data.RFF)_$(Param_Data.nbSignals)_$(Param_Data.name)"
    else 
        savename ="$(Param_Data.E)_$(Param_Data.S)_$(Param_Data.C)_$(Param_Data.RFF)_$(Param_Data.nbSignals)_$(Param_Data.name)_$(Param_Data.Augmentation_Value.Channel)_$(Param_Data.Augmentation_Value.Channel_Test)_nbAugment_$(Param_Data.Augmentation_Value.nb_Augment)"
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
        @pgf push!(a, LegendEntry("Seed $(i)")) # Train)
      
    end
    pgfsave("$(savepathbson)/F1_Score_$(savename).tex",a)

end




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
    @infiltrate
    pgfsave("$(savepathbson)/F1_Score_$(savename).tex",a)

end



function F1_score_TPG_CPU_GPU(Param_Data,Param_Network,Table_Seed_Network)
    hardware1 = "GPU"
    hardware2 ="CPU"
    hardware3 ="TPG"

    @infiltrate
    savepathbson = "run/WiSig/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)_$(Param_Network.Networkname)/$(Param_Data.txs)_$(Param_Data.rxs)/$(Param_Data.txs)_$(Param_Data.rxs)_$(Param_Data.days)_$(Param_Data.equalized)_$(Param_Data.nbSignals)"

    savepathbsonGPU = "run/WiSig/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)_$(Param_Network.Networkname)/$(Param_Data.txs)_$(Param_Data.rxs)/$(Param_Data.txs)_$(Param_Data.rxs)_$(Param_Data.days)_$(Param_Data.equalized)_$(Param_Data.nbSignals)/$(hardware1)"
    savepathbsonCPU = "run/WiSig/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)_$(Param_Network.Networkname)/$(Param_Data.txs)_$(Param_Data.rxs)/$(Param_Data.txs)_$(Param_Data.rxs)_$(Param_Data.days)_$(Param_Data.equalized)_$(Param_Data.nbSignals)/$(hardware2)"
    savepathbsonTPG = "TPG/WiSig_Original/run/ManySig"

    savename ="$(Param_Data.txs)_$(Param_Data.rxs)_$(Param_Data.days)_$(Param_Data.equalized)_$(Param_Data.nbSignals)"
  
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


    for i =1 :1: size(Table_Seed_Network,1)
        Param_Network.Seed_Network= Table_Seed_Network[i]
        name = savename
        F1GPU ="$(savepathbsonGPU)/F1_Score_$(hardware1)_seed_$(Param_Network.Seed_Network)_dr$(Param_Network.Train_args.dr).csv"
        F1CPU ="$(savepathbsonCPU)/F1_Score_$(hardware2)_seed_$(Param_Network.Seed_Network)_dr$(Param_Network.Train_args.dr).csv"
        F1TPG ="$(savepathbsonTPG)/Release_$(Param_Data.txs)_$(Param_Data.rxs)_$(Param_Data.days)_$(Param_Data.equalized)_$(Param_Network.Seed_Network)/F1Score.csv"
    
        delim=';'
        nameBase = split(F1GPU,".")[1]
        #  ----------------------------------------------------
        # --- Loading the matrix 
        # ----------------------------------------------------- 
        matrice_CPU = Matrix(DataFrame(CSV.File(F1CPU;delim,types=Float64,header=false)))
        Score_CPU=zeros(size(matrice_CPU,1),5)

        Score_CPU[:,1]=matrice_CPU[:,1]
        Score_CPU[:,2]=matrice_CPU[:,2]
        Score_CPU[:,3]=matrice_CPU[:,3]
        Score_CPU[:,4]=matrice_CPU[:,4]
        Score_CPU[:,5]=matrice_CPU[:,5]


        matrice_GPU = Matrix(DataFrame(CSV.File(F1GPU;delim,types=Float64,header=false)))
        Score_GPU=zeros(size(matrice_GPU,1),5)

        Score_GPU[:,1]=matrice_GPU[:,1]
        Score_GPU[:,2]=matrice_GPU[:,2]
        Score_GPU[:,3]=matrice_GPU[:,3]
        Score_GPU[:,4]=matrice_GPU[:,4]
        Score_GPU[:,5]=matrice_GPU[:,5]


        matrice_TPG = Matrix(DataFrame(CSV.File(F1TPG;delim,types=Float64,header=false)))
        Score_TPG=zeros(size(matrice_TPG,1),5)

        Score_TPG[:,1]=matrice_TPG[:,1]
        Score_TPG[:,2]=matrice_TPG[:,2]
      #  Score_TPG[:,3]=matrice_TPG[:,3]
      #  Score_TPG[:,4]=matrice_TPG[:,4]
      #  Score_TPG[:,5]=matrice_TPG[:,5]
       
        @pgf push!(a,Plot({color=dictColor[i],mark=dictMarker[i]},Table([(Score_GPU[:,1].-Score_GPU[1,1]),Score_GPU[:,2]])))
        @pgf push!(a, LegendEntry("GPU")) # Train)

        @pgf push!(a,Plot({color=dictColor[2],mark=dictMarker[i]},Table([(Score_CPU[:,1].-Score_CPU[1,1]),Score_CPU[:,2]])))
        @pgf push!(a, LegendEntry("CPU")) # Train)
        @pgf push!(a,Plot({color=dictColor[3],mark=dictMarker[i]},Table([((Score_TPG[:,1].-Score_TPG[1,1]))/1000,Score_TPG[:,2]])))
        @pgf push!(a, LegendEntry("TPG")) # Train)
      
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
    
    savepath = "../My_RFFI_Syst/run/$(Param_Data.Augmentation_Value.augmentationType)_$(Param_Data.nbTx)_$(Param_Data.Chunksize)_$(Param_Network.Networkname)/$(Param_Data.E)_$(Param_Data.S)"
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
    
   #=
    Scenario_5 ="$(savepath)/$(name5)/$(hardware1)/F1_Score_$(hardware1)_seed_$(Param_Network.Seed_Network).csv"
   # Scenario_3 ="$(savepath)/$(name3)/$(hardware1)/F1_Score_$(hardware1).csv"
   # Scenario_2 ="$(savepath)/$(name2)/$(hardware1)/F1_Score_$(hardware1).csv"
   # Scenario_1 ="$(savepath)/$(name1)/$(hardware1)/F1_Score_$(hardware1).csv"
   # Scenario_05 ="$(savepath)/$(name05)/$(hardware1)/F1_Score_$(hardware1).csv"

    delim=';'
    nameBase = split(Scenario_5,".")[1]
    #  ----------------------------------------------------
    # --- Loading the matrix 
    # ----------------------------------------------------- 
    # --- Load the matrix  as float64
    matrice_5 = Matrix(DataFrame(CSV.File(Scenario_5;delim,types=Float64,header=false)))
    Score_5=zeros(size(matrice_5,1),5)
    Score_5[:,1]=matrice_5[:,1]# temps 
    Score_5[:,2]=matrice_5[:,2]
    Score_5[:,3]=matrice_5[:,3]
    Score_5[:,4]=matrice_5[:,4]
    Score_5[:,5]=matrice_5[:,5]

    matrice_3 = Matrix(DataFrame(CSV.File(Scenario_3;delim,types=Float64,header=false)))
    Score_3=zeros(size(matrice_3,1),3)
    Score_3[:,1]=matrice_3[:,1]# temps 
    Score_3[:,2]=matrice_3[:,2]
    Score_3[:,3]=matrice_3[:,3]

    matrice_2 = Matrix(DataFrame(CSV.File(Scenario_2;delim,types=Float64,header=false)))
    Score_2=zeros(size(matrice_2,1),3)
    Score_2[:,1]=matrice_2[:,1]# temps 
    Score_2[:,2]=matrice_2[:,2]
    Score_2[:,3]=matrice_2[:,3]

    matrice_1 = Matrix(DataFrame(CSV.File(Scenario_1;delim,types=Float64,header=false)))
    Score_1=zeros(size(matrice_1,1),3)
    Score_1[:,1]=matrice_1[:,1]# temps 
    Score_1[:,2]=matrice_1[:,2]
    Score_1[:,3]=matrice_1[:,3]

    matrice_05 = Matrix(DataFrame(CSV.File(Scenario_05;delim,types=Float64,header=false)))
    Score_05=zeros(size(matrice_05,1),3)
    Score_05[:,1]=matrice_05[:,1]# temps 
    Score_05[:,2]=matrice_05[:,2]
    Score_05[:,3]=matrice_05[:,3]

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
                Plot({color=dictColor[1],mark=dictMarker[1]},Table([(Score_5[:,1].-Score_5[1,1]),Score_5[:,2]])),              # Cusrto marker and color and Table to plot the object with first column X and second column Y
                LegendEntry("$(name5) "), # Train

                Plot({color=dictColor[1],mark=dictMarker[1]},Table([(Score_5[:,1].-Score_5[1,1]),Score_5[:,5]])),              # Cusrto marker and color and Table to plot the object with first column X and second column Y
                LegendEntry("$(name5) Test"), # Train

            #    Plot({color=dictColor[2],mark=dictMarker[1]},Table([(Score_3[:,1].-Score_3[1,1]),Score_3[:,2]])),              # Cusrto marker and color and Table to plot the object with first column X and second column Y
            #    LegendEntry("$(name3) "), # Train

            #    Plot({color=dictColor[3],mark=dictMarker[1]},Table([Score_2[:,1],Score_2[:,2]])),              # Cusrto marker and color and Table to plot the object with first column X and second column Y
            #    LegendEntry("$(name2) "), # Train

            #    Plot({color=dictColor[4],mark=dictMarker[1]},Table([Score_1[:,1],Score_1[:,2]])),              # Cusrto marker and color and Table to plot the object with first column X and second column Y
            #    LegendEntry("$(name1) "), # Train

            #    Plot({color=dictColor[5],mark=dictMarker[1]},Table([Score_05[:,1],Score_05[:,2]])),              # Cusrto marker and color and Table to plot the object with first column X and second column Y
            #    LegendEntry("$(name05) "), # Train
                );
    pgfsave("$(savepath)/F1_Score_$(name).tex",a)
    =#    
end 


=#
end 