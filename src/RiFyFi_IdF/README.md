# RiFyFi_IdF

[![Build Status](https://github.com/achillet/RiFyFi_IdF.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/achillet/RiFyFi_IdF.jl/actions/workflows/CI.yml?query=branch%3Amain)


The Module RiFyFi_IdF defines function related to the classification/ Identification Framework particularly based on CNN
Four different CNN are impleted here but you can implement your own NN.

The file Struct_Network.jl defines a structure to save information concerning the network: name, leraning parameters, numbers of epoch ... Feel free to add any parameter you want to change.


function initAlexNet(x)
    - definition des différentes couche du reseaux
    - définition de la crossentropy 
    return (m,loss)
end


# customTrain.jl

function customTrain!(nn,customLoss,dataTrain,dataTest,savepath;kws...)
    - décision GPU ou CPU
    - on charge le model sur le hardware choisi
    - apprentissage : à chaque epoch on lance le timer et les données sont présentées au réseau. 
    Un Chunk (séquence de 256 IQ) est présenté et passe dans le réseau, puis une autre jusqu'a atteindre batchsize. Lorsque le nombre de Chunk présenté atteint batchsize le réseau est mis à jour par backpropagation et on passe au batch suivant. Une fois que tout les batch ont été parcouru on a atteint une epoch, on va tester l'accuracy du reseau en Test et en Train et afficher les résultats. 
    - Après chaque epoque le Timer est arrété et le score F1 est calculé par la fonction F1Score 
return nn,trainLoss,trainAcc,testLoss,testAcc,args



function F1Score(loader,model,device=cpu)
    - calcul le F1 score global à partir des estimations de classe et les classes vraies.
return F1
