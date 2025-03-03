module Palette
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


using InteractiveUtils: clipboard
using Infiltrator
include("../../LatexConfusionMatrix/src/LatexConfusionMatrix.jl")
using .LatexConfusionMatrix

export Table_colorLatex

PREAMBULE_LATEX = ["
\\documentclass[landscape]{standalone}
\\usepackage[T1]{fontenc}
\\usepackage[utf8]{inputenc}
\\usepackage{xcolor}
\\usepackage{colortbl}
\\usepackage{diagbox}
%\\usepackage[landscape]{geometry}
"]



function Table_colorLatex(CSV_file)
    @infiltrate
    nbdays=4
    header      = (["day\$_$i\$" for i = 1 : 1 : nbdays])
    row_labels  = (["day\$_$i\$" for i = 1 : 1 : nbdays])
    row_label_column_title = "\\backslashbox[15mm][]{\\footnotesize Train}{\\footnotesize Test}"
    cellcolor_header = "blue!10!white"      
    delim='&'
    standalone=true
    doClipboard = false

    nameBase = split(CSV_file,".")[1]
    #  ----------------------------------------------------
    # --- Loading the matrix 
    # ---------------------------------------------------- 
    # --- Load the matrix  as float64
    m = 100 .- Matrix(DataFrame(CSV.File(CSV_file;delim,types=Float64,header=false)))

    hg = LatexHighlighter((m,i,j)-> true, red_grad_formatter)

    if !isempty(cellcolor_header)
        tf = PrettyTables.LatexTableFormat(;header_envs=["cellcolor{$cellcolor_header}"])
    else 
        tf = PrettyTables.LatexTableFormat(;header_envs=[])
    end
    open("$nameBase.tex","w") do io 
        if standalone
            # Write the preambule 
            for tr in PREAMBULE_LATEX
                write(io,tr)
            end
            write(io,"\n\\begin{document}\n")
        end
        # Write the table 
        pretty_table(io,m, backend = Val(:latex); header=LatexCell.(header),row_labels,highlighters = (hg),alignment=:c,tf,row_label_column_title,row_label_alignment=:c)
        if standalone
            # End of document
            write(io,"\n\\end{document}")
        end
    end
end 




function CreatePalette()

 #   dictMarker  = ["square*","triangle*","diamond*","*","pentagon*","rect","otimes","triangle*"];
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

    
    for i =1 :1: size(dictColor)
       
        Score_5=zeros(size(dictColor),5)

        Score_5[:,1] .= 1
        Score_5[:,2] .= 2
       
        @pgf push!(a,Plot({color=dictColor[i]},Table([Score_5[:,1],Score_5[:,2]])))
        @pgf push!(a, LegendEntry("Seed $(i)")) # Train)
      
    end
    pgfsave("./Palette.tex",a)

end




end 