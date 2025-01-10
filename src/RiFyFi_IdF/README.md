# RiFyFi_IdF

[![Build Status](https://github.com/achillet/RiFyFi_IdF.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/achillet/RiFyFi_IdF.jl/actions/workflows/CI.yml?query=branch%3Amain)

The RiFyFi_IdF module defines functions related to the classification/identification framework, specifically based on CNN.
Four different CNN are implemented here, but you can implement your own CNN.

The file Struct_Network.jl defines a structure to store information about the network: name, learning parameters, epoch numbers ... Feel free to add any parameter you want to change.

function initAlexNet(x)
    - define the different layers of the network
    - define the crossentropy 
    return (m,loss)
end

# RiFyFi_IdF.jl 
This file describes different network architectures, you can easily create your own architecture following the same functional structure and use it with the global framework.

# customTrain.jl

function customTrain!(nn,customLoss,dataTrain,dataTest,savepath;kws...)
    - Decide whether to use GPU or CPU
    - Upload the model to the hardware
    - Training: The network is fed with data. A chunk (sequence of 256 IQ samples) is given to the input of the network. When the number of chunks corresponds to the batch size, the network is updated by backpropagation. It's then possible to continue with the second batch. An epoch ends when all the batch are seen by the network. Then the accuracy of the network is evaluated in train and test. 
    - apprentissage : à chaque epoch on lance le timer et les données sont présentées au réseau. 
    Un Chunk (séquence de 256 IQ) est présenté et passe dans le réseau, puis une autre jusqu'a atteindre batchsize. Lorsque le nombre de Chunk présenté atteint batchsize le réseau est mis à jour par backpropagation et on passe au batch suivant. Une fois que tout les batch ont été parcouru on a atteint une epoch, on va tester l'accuracy du reseau en Test et en Train et afficher les résultats. 
return nn,trainLoss,trainAcc,testLoss,testAcc,args



function F1Score(loader,model,device=cpu)
    - calcul le F1 score global à partir des estimations de classe et les classes vraies.
return F1
