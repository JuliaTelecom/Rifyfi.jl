# Pretty color/marker
function latexUtils()
    counterMarker = [1]
    counterColor  = [1]
	dictMarker  = ["square*","triangle*","diamond*","*","pentagon*","rect","otimes","triangle*"];
    lMarker     = length(dictMarker)
    dictColor   = ColorSchemes.tableau_superfishel_stone
    lColor       = length(dictColor)
    function update()
        marker = dictMarker[mod(counterMarker[1] - 1, lMarker) + 1]
        color  = dictColor[mod(counterColor[1] - 1, lColor) + 1]
        counterMarker[1] += 1 
        counterColor[1] +=1 
        return (mark=marker,color=color)
    end
    return update
end


function PlotingAMAM_Saleh(x,y,RFF,r,E,S,C,name="control")
    
p_latex = latexUtils()
dictColor   = ColorSchemes.tableau_superfishel_stone

tup = p_latex()

savepath= "../My_Signal_Synthesis/Plots/$(E)_$(S)_$(C)_$(RFF)_$(name)"
!ispath(savepath) && mkpath(savepath)

# Calculate AM/PM as difference in phase with modulo shenaniggans
am_pm = mod.(angle.(y)-angle.(x),2π)


@pgf a = PGFPlotsX.Axis({
                         height      ="3in",
                         width       ="4in",
                         grid,
                         xlabel      = "PA input magnitude (value)",
                         ylabel      = "PA Output mangitude (value)",
                         title       = "Saleh AM/AM",
                         legend_style="{at={(0,1)},anchor=north west,legend cell align=left,align=left,draw=white!15!black}"
                        },
                        Plot({"only marks",color=dictColor[r],mark=tup.mark},Table([abs.(x),abs.(y)]))
                       );
                        pgfsave("$(savepath)/SalehAMAM_$(r).tex",a)

tup = p_latex()
@pgf b= PGFPlotsX.Axis({
                         height      ="3in",
                         width       ="4in",
                         grid,
                         xlabel      = "PA input magnitude (value)",
                         ylabel      = "PA output phase distortion (rad)",
                         title       = "Saleh AM/PM",
                         legend_style="{at={(0,1)},anchor=north west,legend cell align=left,align=left,draw=white!15!black}"
                        },
                       Plot({"only marks",color=dictColor[r],mark=tup.mark},Table([abs.(x),am_pm]))
                       );
                       pgfsave("$(savepath)/SalehAMPM_$(r).tex",b)



# Figure for first report page
p_latex = latexUtils()  # Instantiate counter 
tup = p_latex()         # We do not want the first 
tup = p_latex()         # Second couple for first curve 
tup2 = p_latex()        # Third couple for second curbe 
@pgf c= PGFPlotsX.Axis({
                         height      ="3in",
                         width       ="4in",
                         grid,
                         xlabel      = "PA input magnitude (value)",
                         ylabel      = "PA output distortion",
                         # title       = "Saleh AM/PM",
                         legend_style="{at={(0,1)},anchor=north west,legend cell align=left,align=left,draw=white!15!black}"
                        },
                        Plot({"only marks",color=tup.color,mark=tup.mark},Table([abs.(x),abs.(y)])),
                        LegendEntry("AM/AM"),
                       Plot({"only marks",color=tup2.color,mark=tup2.mark},Table([abs.(x),am_pm])),
                        LegendEntry("AM/PM"),
                       );
                       pgfsave("$(savepath)/SalehAMPMbis_$(r).tex",c)



end 