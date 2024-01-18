module LatexConfusionMatrix

# ----------------------------------------------------
# --- Dependencies 
# ---------------------------------------------------- 
using DataFrames, CSV       # Manage confusion matrix files 
using PrettyTables          # Rendering latex 
using Colors, ColorSchemes  # Gradient colors
using InteractiveUtils: clipboard
using Infiltrator
# --- Methods extension 
import Base:getindex

# ----------------------------------------------------
# --- Exportation 
# ---------------------------------------------------- 
export latexConfusionMatrix
export good_grad_formatter
export bad_grad_formatter

# ----------------------------------------------------
# --- Standalone Latex mode,
# ---------------------------------------------------- 
# This is what will be pushed in the beginning of the latex file
# Additional packages can be added with push!(PREAMBULE_LATEX,"\\usepackage{mypackage}"
PREAMBULE_LATEX = ["
\\documentclass[landscape]{standalone}
\\usepackage[T1]{fontenc}
\\usepackage[utf8]{inputenc}
\\usepackage{xcolor}
\\usepackage{colortbl}
\\usepackage{diagbox}
%\\usepackage[landscape]{geometry}
"]

# ----------------------------------------------------
# --- Tools for color managment 
# ---------------------------------------------------- 
""" Convert a RGB object to something manageable in text 
"""
function rgbLatex(color::RGB)
    # Extract the object as a tuple of color, each of them convert as a float 
    return (rgbLatex(color.r),rgbLatex(color.g),rgbLatex(color.b))
end
function rgbLatex(r::Number)
    # Convert the UInt representation into a representation between 0 and 1
    return round(round(Int,r*255)/255;digits=3)
end

# ----------------------------------------------------
# --- Cell formatters
# ---------------------------------------------------- 
# const goodGrad = ColorScheme(range(colorant"red", colorant"green", length=100))
""" A custom structure for color gradient, color application and range application 
"""
struct CustomGrad{T} 
    isNothing::Bool         # Apply a colorsheme
    colormap::Array{RGB{T}}    # The grid 
    minVal::Float64         # Minimal value where the color is applied 
    maxVal::Float64         # Maximal value where the color is applied 
end
# Constructors 
customGrad() = CustomGrad(true,Array{RGB}(undef,0),0.0,0.0)
customGrad(colormap::Array{RGB{T}},minVal,maxVal) where T = CustomGrad(false,colormap,minVal,maxVal)

""" Extract appropriate color based on index and color objet 
"""
function getindex(obj::CustomGrad{T},val) where T
    # Interval to cover with colors 
    # Convert the value we want into a scale between 0 and 1 
    posInRange = (val - obj.minVal) / (obj.maxVal - obj.minVal)
    # Create a scale for the color, also between 0 and 1 
    colorIn = range(0;stop=1,length=length(obj.colormap))
    # Best color is the one that minimize distance between the 2 ranges 
    # Saturation with lower and maximal color 
    pos = findmin(abs.( posInRange .- colorIn))[2]
    return obj.colormap[pos]
end

# These functions are to color the background of the cell, based on its value 
""" Fill with appropriate color, based on grad
"""
function formatter(obj::CustomGrad,m,i,j,v)
    if obj.isNothing
        # No gradient is given, returns a normal cell
        return v
    else 
        # We have a grad, color it
        pos = 1+Int(floor(m[i,j]))
        tt= obj[pos]
        (rr,gg,bb) = rgbLatex(tt)        # Convert in rgb
        return "\\cellcolor[rgb]{$rr,$gg,$bb}{\\textcolor{black}{$v}}"
    end
end 

""" Fill the cell with appropriate color for good scores.  
"""
function good_grad_formatter(m,i,j,v)
    grad        = colormap("Greens",30)[1:20] # Only beginning to avoid dark colors
    obj         = customGrad(grad,50.0,100.0) 
    return formatter(obj,m,i,j,v)
end

function bad_grad_formatter(m,i,j,v)
    grad        = colormap("grays",30)[1:20] # Only beginning to avoid dark colors
    obj         = customGrad(grad,0.0,25.0) 
    return formatter(obj,m,i,j,v)
end

""" Load the confusion matrix stored in `file` and create a Latex table 
Optional arguments
- doSave : save the latex table to `file.tex` (default `no` )
- delim : delimiter of the CSV file (default ;)
- standalone : create a compilable latex file (only valid with doSave = true)
- headers : Vector of string of table header (column names)
- row_labels : Vector of strings for name of rows 
- doClipboard : Save the table in clipboard (default false, only valid with doSave = false)
- row_name_column_title = String to put in the left hand corner of the table (split row/header)
- cellcolor_header = "" An additional highlighter for cell header(for instance cellcolor_header="yellow!10!white")
"""
function latexConfusionMatrix(file::String;doSave::Bool = false,delim=';',standalone=false,header,row_labels,doClipboard = false,row_label_column_title="",cellcolor_header="")
    
    # --- Get the filename without the extension 
    nameBase = split(file,".")[1]
    #  ----------------------------------------------------
    # --- Loading the matrix 
    # ---------------------------------------------------- 
    # --- Load the matrix  as float64
    m = Matrix(DataFrame(CSV.File(file;delim,types=Float64,header=false)))
    # ----------------------------------------------------
    # --- Defines formatter 
    # ---------------------------------------------------- 
    # Diagonal formatter : Green is good 
    hg = LatexHighlighter((m, i, j) -> (i==j), good_grad_formatter)
    # No diagonal, gray  is bad
    hb = LatexHighlighter((m, i, j) -> (i!=j), bad_grad_formatter)
    # Specific header formatter 
    if !isempty(cellcolor_header)
        tf = PrettyTables.LatexTableFormat(;header_envs=["cellcolor{$cellcolor_header}"])
    else 
        tf = PrettyTables.LatexTableFormat(;header_envs=[])
    end
    # ----------------------------------------------------
    # --- Create latex file 
    # ---------------------------------------------------- 
    if doSave 
        # --- Save the table in a tex file 
        open("$nameBase.tex","w") do io 
            if standalone
                # Write the preambule 
                for tr in PREAMBULE_LATEX
                    write(io,tr)
                end
                write(io,"\n\\begin{document}\n")
            end
            # Write the table 
            pretty_table(io,m, backend = Val(:latex); header=LatexCell.(header),row_labels,highlighters = (hg,hb),alignment=:c,tf,row_label_column_title,row_label_alignment=:c)
            if standalone
                # End of document
                write(io,"\n\\end{document}")
            end
        end
        return nothing
    else
        # Just print the result on the REPL
        io = IOBuffer();
        pretty_table(io,m, backend = Val(:latex); header=LatexCell.(header),row_labels,highlighters = (hg,hb),alignment=:c,tf,row_label_column_title,row_label_alignment=:c) |> clipboard
        # println(io)
        theTable =  String(take!(io))
        if doClipboard
            clipboard(theTable)
        end
        # print(theTable)
        return println(theTable)
    end
end



end
