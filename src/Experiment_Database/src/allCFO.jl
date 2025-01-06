# ----------------------------------------------------
# --- Dependencies 
# ----------------------------------------------------
#using AbstractSDRsUtils # Load / store  
using DSP               # For correlation // FFT 
using PGFPlotsX         # Latex rendering 
using ColorSchemes      # Color switch 
using KissSmoothing     # Smoothing CFO curve  
using Plots

function applyCFO(sig, ΔF, Fe)
    n = 0:length(sig)-1
    return sig .* exp.(1im * 2 * π * ΔF * n / Fe)
end

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
 

# --- Loading ideal signal 
# Serve as baseline for Γ
sigId = readComplexBinary("/media/redinblack/ANR_RedInBlack/rffExperiment/binaryDataInput/Preamble.dat")
nFFT = 512 
nCP = 36
sizeSymb = (nFFT+nCP) * 30
Fe = 4e6

function calculate_cfo(filePath,filename,ident_radio)
    sigTx = readComplexBinary("$filePath/$filename")
 #   sigTx = applyCFO(sigTx, -3000, Fe)
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

    # --- Sync the 2 signals 
    # We should have a max in the first preamble 
    # If this does not work, taht will only shift the initial value of the CFO (sync error is rotation in freq. domain) 
    theCorr = xcorr(sigTx,sigId)
    indexBeg = argmax(abs2.(theCorr[1:sizeSymb]))
    pp = plot(abs.(theCorr[(indexBeg) .+ (-4000:4000)])) 

    # Number of symbols in the sequence 
    nbSymb  = length(sigTx[indexBeg:end]) ÷ sizeSymb
    sigUsed = sigTx[indexBeg .+ (1:nbSymb * sizeSymb)]
    ϕVect = zeros(Float64,nbSymb)
    for n ∈ 1:nbSymb 
        # Get symbol of interest 
        tmp = sigUsed[ (n-1)*sizeSymb .+ (1:sizeSymb)]
        # Calculate autocorr assuming perfect sync. 
        Γ   = 1/sizeSymb * sum( tmp .* conj(sigId))
        # Calculate argument 
        # If no CFO, Γ is real so CFO is in the phase 
        ϕ   = angle(Γ)
        ϕVect[n] = ϕ
    end 
    # Actual CFO is only difference in phase between 2 consecutive symbols 
    # As we has a preamble size between 2 consecutive measures we can deduce the phase advance by calculating how it should have rotated
    if ident_radio == "BladeRF"
        cfoVect = diff(unwrap(ϕVect)) / (2π * sizeSymb) * Fe .+ 2500
    elseif ident_radio == "Pluto"
        cfoVect = diff(unwrap(ϕVect)) / (2π * sizeSymb) * Fe .+ 7700
    elseif ident_radio == "e310rose2"
        cfoVect = diff(unwrap(ϕVect)) / (2π * sizeSymb) * Fe .+ 3000
    elseif ident_radio == "e310rose4"
        cfoVect = diff(unwrap(ϕVect)) / (2π * sizeSymb) * Fe .+ 1700
    elseif ident_radio == "x310"
        cfoVect = diff(unwrap(ϕVect)) / (2π * sizeSymb) * Fe .+ 500
    end
    Ts = sizeSymb / Fe
    timeVect = (0:length(cfoVect)-1) * Ts
    return timeVect, cfoVect
end



@pgf a = Axis({
height      ="3in",
width       ="4in",
grid,
xlabel      = "Time [s]",
ylabel      = "Frequency [Hz]",
legend_style="{at={(1,1)},anchor=north east,legend cell align=left,align=left,draw=white!15!black}"
},
);



# --- Loading measured signal
filePath = "/media/redinblack/ANR_RedInBlack/rffExperiment/2024-06-03-17h/Preamble/DatFile_Cut/"
filePath = "/media/redinblack/ANR_RedInBlack/rffExperiment/2024-06-12-17h/Preamble/DatFile_Cut/"

plt = plot();
for f in readdir(filePath)
    if occursin(".dat",f)
        # Get radio name 
        ident_radio = split(f,"_")[1]
        # Calculate CFO vs Time 
        timeVect,cfoVect = calculate_cfo(filePath,f,ident_radio)
        # SMooth the curve 
        cfoS,_ = denoise(cfoVect;factor=0.6)
        # Get new color and plot
        tup = updateLatex()
        @pgf push!(a,Plot({tup...},Table(timeVect,cfoS)))
        @pgf push!(a,LegendEntry(ident_radio))
    end
end
display(a)

