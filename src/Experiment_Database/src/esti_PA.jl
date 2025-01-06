using AbstractSDRsUtils 
using DSP 
# using Plots 
# plotlyjs()
using PGFPlotsX
using ColorSchemes

function latexUtils()
    counterMarker = [1]
    counterColor  = [1]
    dictMarker  = ["square*","triangle*","diamond*","*","pentagon*","triangle*","halfcircle*","halfsquare right*"];
    lMarker     = length(dictMarker)
    dictColor   = ColorSchemes.tableau_superfishel_stone
    lColor       = length(dictColor)
    function update()
        marker = dictMarker[mod(counterMarker[1] - 1, lMarker) + 1]
        color  = dictColor[mod(counterColor[1] - 1, lColor) + 1]
        counterMarker[1] += 1
        counterColor[1] +=1
        #return PGFPlotsX.Options(Dict(:mark=>marker,:color=>color)...)
        return PGFPlotsX.Options(Dict(:color=>color,:line_width=>1)...)
    end
    return update
end
updateLatex = latexUtils()

function applyCFO(sig, ΔF, Fe)
    n = 0:length(sig)-1
    return sig .* exp.(1im * 2 * π * ΔF * n / Fe)
end

# --- Core parameters 
Fe = 4e6

# --- Loading ideal signal 
sigId = readComplexBinary("/media/redinblack/ANR_RedInBlack/rffExperiment/binaryDataInput/Preamble.dat")
nFFT = 512 
nCP = 36
nnS = 30
sizeSymb = (nFFT+nCP) * nnS

# --- Loading mesaured signal


function get_AM_AM(filePath,filename)
    sigTx = readComplexBinary("$(filePath)/$(filename)")
    sigTx = applyCFO(sigTx, -2500, Fe)

    # --- Sync the 2 signals 
    # We should have a max in the first preamble 
    # If this does not work, taht will only shift the initial value of the CFO (sync error is rotation in freq. domain) 
#  theCorr = xcorr(sigTx,sigId)
#  indexBeg = argmax(abs2.(theCorr[1:sizeSymb]))
#    y = sigTx[indexBeg .+ (1:(nFFT+nCP)*nnS)]
#    x = sigId
#    pp = plot(abs.(theCorr[indexBeg-10000:indexBeg+10000]))
#    display(pp)


    theCorr = xcorr(sigTx,sigId[1:(nFFT+nCP)*30])
    indexBeg = argmax(abs2.(theCorr[1:1000*sizeSymb]))
    y = sigTx[indexBeg .+ (1:(nFFT+nCP)*nnS)]
    x = sigId

 #   pp =plot(abs.(theCorr[(indexBeg) .+ (-4000:4000)])) 


    return (x,y)
end 

@pgf a = PGFPlotsX.Axis({
                         height      ="3in",
                         width       ="4in",
                         ymax = 0.1,
                         grid,
                         xlabel      = "PA input magnitude (value)",
                         ylabel      = "PA Output mangitude (value)",
                         title       = "Saleh AM/AM",
                         legend_style="{at={(0,1)},anchor=north west,legend cell align=left,align=left,draw=white!15!black}"
                        },
)

filePath = "/media/redinblack/ANR_RedInBlack/rffExperiment/2024-06-12-17h/Preamble/DatFile_Cut/"


filePath = "/media/redinblack/ANR_RedInBlack/rffExperiment/2024-07-04-14h/Preamble/DatFile_Cut/"
plt = plot();
for f in readdir(filePath)
    if occursin(".dat", f)
        # Get radio name 
        ident_radio = split(f, "_")[1]
        sigTx = readComplexBinary("$(filePath)/$(f)")
        if ident_radio == "BladeRF"
            sigTx = applyCFO(sigTx, -2500, Fe)
        elseif ident_radio == "Pluto"
            sigTxt = applyCFO(sigTx, -7700, Fe)
        elseif ident_radio == "e310rose2"
            sigTx = applyCFO(sigTx, -3000, Fe)
        elseif ident_radio == "e310rose4"
            sigTx = applyCFO(sigTx, -1700, Fe)
        elseif ident_radio == "x310"
            sigTx = applyCFO(sigTx, -500, Fe)
        end


        theCorr = xcorr(sigTx,sigId[1:(nFFT+nCP)*30])
        indexBeg = argmax(abs2.(theCorr[1:1000*sizeSymb]))
        pp =plot(abs.(theCorr[(indexBeg) .+ (-4000:4000)])) 

        y = sigTx[indexBeg .+ (1:(nFFT+nCP)*nnS)]
        x = sigId
      #  (x,y) = get_AM_AM(filePath,f)
        # Get new color and plot
        tup = updateLatex()
        @pgf push!(a, Plot({"only marks",tup...}, Table(abs.(x), abs.(y))))
        @pgf push!(a, LegendEntry(ident_radio))
    end
end
display(a)
