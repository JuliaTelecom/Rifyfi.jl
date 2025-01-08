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





function MainPlottingMatrix_Latex(file,nbRadioTx=4)
    # --- Create the header
    header      = (["Tx\$_$i\$" for i = 1 : 1 : nbRadioTx])
    row_labels  = (["TxTrue\$_$i\$" for i = 1 : 1 : nbRadioTx])
    row_label_column_title = "\\backslashbox[15mm][]{\\footnotesize True}{\\footnotesize Guess}"
    cellcolor_header = "blue!10!white"      
    m = latexConfusionMatrix(file;doSave=true,standalone=true,header,row_label_column_title,row_labels,cellcolor_header)
end 