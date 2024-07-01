using Pickle 


export Pickle_to_matrix
"""Creer une matrice bigMat pour des Txs, Rxs, jours, equalized or not donnés \n
#   Parametres sous la forme de liste 1:6, 1:12; 1:4, 1:2 et 1 ou 2  || [,] 
"""
function Pickle_to_matrix(Param_Data)
    # --- Load file 
    
    obj = myunpickle(Param_Data.File_name)    
    # --- Estimate the number of burst for train and test
    nbBurstTotal    = evaluate_total_number_of_burst_Spec(obj,Param_Data.txs,Param_Data.rxs,Param_Data.days,Param_Data.equalized) 
    nbBurstTotalTrain = Int(nbBurstTotal*Param_Data.pourcentTrain) 
    nbBurstTotalTest = nbBurstTotal - nbBurstTotalTrain
    # --- Create and init temporary matrix 
    bigMatTemp = zeros(Float32,Param_Data.Chunksize,2,nbBurstTotal) 
    bigLabelsTemp   = zeros(Int,Param_Data.nbTx,nbBurstTotal) 
    # --- Create and init Data and labels matrix 
    bigMatTrain      = zeros(Float32,Param_Data.Chunksize,2,nbBurstTotalTrain)
    bigMatTest      = zeros(Float32,Param_Data.Chunksize,2,nbBurstTotalTest)
    bigLabelsTrain   = zeros(Int,Param_Data.nbTx,nbBurstTotalTrain) 
    bigLabelsTest   = zeros(Int,Param_Data.nbTx,nbBurstTotalTest) 
    tmp_somme = 0
    tmp_sommeTrain = 0
    tmp_sommeTest = 0
    # --- browse all of the options 
    for tx in Param_Data.txs
        for rx in Param_Data.rxs 
            for day in Param_Data.days
                for eq in Param_Data.equalized
                    # -- Shared each groups of signal between Train and Test 
                    # --- Load Data and determine number of burst ()
                    data::Array{Float32}   = obj["data"][tx][rx][day][eq]
                    (nbBurst,_,_) = size(data)
                    nbBurstTrain = Int( nbBurst * Param_Data.pourcentTrain)
                    nbBurstTest = nbBurst - nbBurstTrain
                    # ici il faut diviser on aura nbBurst à mult par pourcentrain pour savoir combien on prend de Burst en train pour cette config (TX, rx,jours)
                    if nbBurst != 0
                        # On utilise la fonction create_bigMat_ManySig pour créer des vecteurs temporaire qu'on vient diviser pour Train et Test 
                        (bigMatTemp[:,:,tmp_somme+1:tmp_somme+nbBurst],bigLabelsTemp[:,tmp_somme+1:tmp_somme+nbBurst]) = create_bigMat_ManySig(data,tx,Param_Data.nbTx,Param_Data.Chunksize)    # On sait que 1000 samples
                        bigMatTrain[:,:,tmp_sommeTrain+1:tmp_sommeTrain+nbBurstTrain]  = bigMatTemp[:,:,tmp_somme+1:tmp_somme+nbBurstTrain]#On sait que 1000 samples
                        bigLabelsTrain[:,tmp_sommeTrain+1:tmp_sommeTrain+nbBurstTrain] = bigLabelsTemp[:,tmp_somme+1:tmp_somme+nbBurstTrain]
                        bigMatTest[:,:,tmp_sommeTest+1:tmp_sommeTest+nbBurstTest]   =  bigMatTemp[:,:,tmp_somme+nbBurstTrain+1:tmp_somme+nbBurstTrain+nbBurstTest]
                        bigLabelsTest[:,tmp_sommeTest+1:tmp_sommeTest+nbBurstTest]  =  bigLabelsTemp[:,tmp_somme+nbBurstTrain+1:tmp_somme+nbBurstTrain+nbBurstTest] #On sait que 1000 samples
                        tmp_somme +=nbBurst
                        tmp_sommeTrain +=nbBurstTrain
                        tmp_sommeTest +=nbBurstTest
                    end
                end
            end
        end
    end
    bm::Array{Float32}  = convert(Array{Float32},bigMatTrain)
    bt::Array{Float32} = convert(Array{Float32},bigMatTest)
    lm::Array{Int} = convert(Array{Int},bigLabelsTrain)
    lt::Array{Int} = convert(Array{Int},bigLabelsTest)

        return(bm,bt,lm,lt)
end


""" 
Evalue la taille pour allocation // parametre sous la forme de liste 1:6, 1:12; 1:4, 1:2 et 1 ou 2 
"""
function evaluate_total_number_of_burst_Spec(obj,txs,rxs,days,equalized)
    nbBurst = 0
    for tx ∈ txs
        for rx ∈ rxs
            for d ∈ days
                for eq ∈ equalized
                    data    = obj["data"][tx][rx][d][eq]
                    (tmp,_,_) = size(data)
                    nbBurst += tmp
                end
            end
        end
    end
    return nbBurst
end 

"""Creer une matrice bigMat pour un Tx, un Rx, un jour, equalized or not données"""
function create_bigMat_ManySig(data,tx,nbRadios,ChunkSize)
    cnt = 0
    (nbBurst,_,_) = size(data)
    bigMat      = zeros(Float32,ChunkSize,2,nbBurst)   
    bigLabels   = zeros(Int,nbRadios,nbBurst)       
    if nbBurst != 0
        # --- Recreate data matrix
        theView = @views bigMat[:,:,cnt .+ (1:nbBurst)]
        reshapeData!(theView,data)
        bigLabels[tx , cnt  .+ (1:nbBurst)] .= 1
        # --- Update counter 
        cnt += nbBurst
    end
    return (bigMat,bigLabels)
end

""" Transform Wisig data (nbBurst x ChunkSize x 2) into Oracle compatible format which is ChunkSize x 2 x nbBurst 
"""
function reshapeData!(y,x::Array{Float32})
    nbBurst = size(x,1)
    @assert size(x) == (nbBurst,256,2) "Size of x is incorrect"
    @assert size(y) == (256,2,nbBurst) "Size of y is incorrect"
    for n in 1 : nbBurst 
        y[:,:,n] = Float32.(x[n,:,:])
    end
end

"""Just to check what the .pkl file is """
function myunpickle(filename)
    open(filename,"r") do io 
        Pickle.npyload(io)
    end
end

"""Permet de concatener en le signal I + iQ """
function concaten!(tmp_concat,indexList,bigMat)
    j = 1
    for i in indexList
        tmp_concat[(j-1)*256+1:j*256] =  bigMat[:,:,i][:,1] + bigMat[:,:,i][:,2]im
        j=j+1
    end
end


