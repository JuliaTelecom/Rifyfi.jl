
include("src/Experiment_Database/src/Utils.jl")
using .Utils

include("src/Experiment_Database/src/DatBinaryFiles.jl")
using .DatBinaryFiles




Run1="2024-06-03-17h/"

List_files = readdir("/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run1)Preamble/DatFile_Cut/")

i=1
filename = "/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run1)Preamble/DatFile/$(List_files[i])"
Data_Vector= DatBinaryFiles.readComplexBinary(filename)
gett= Utils.getWaterfall(4e6,Data_Vector)
heatmap(gett)
Cutted_Data_Vector = Data_Vector[1000001:end-5000000]
get= Utils.getWaterfall(4e6,Cutted_Data_Vector)
heatmap(get)
filename_cut= "/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run1)Preamble/DatFile_Cut/$(List_files[i])"
writeComplexBinary(Cutted_Data_Vector,filename_cut)


List_files = readdir("/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run1)Payload/DatFile/")

i=1
filename = "/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run1)Payload/DatFile/$(List_files[i])"
Data_Vector= DatBinaryFiles.readComplexBinary(filename)
get= Utils.getWaterfall(4e6,Data_Vector)
heatmap(get)
Cutted_Data_Vector = Data_Vector[1000001:end-5000000]
get= Utils.getWaterfall(4e6,Cutted_Data_Vector)
heatmap(get)
filename_cut= "/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run1)Payload/DatFile_Cut/$(List_files[i])"
writeComplexBinary(Cutted_Data_Vector,filename_cut)







Run1bis="2024-07-04-14h/"

List_files = readdir("/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run1bis)Preamble/DatFile/")

i=1
filename = "/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run1bis)Preamble/DatFile/$(List_files[i])"
Data_Vector= DatBinaryFiles.readComplexBinary(filename)
get= Utils.getWaterfall(4e6,Data_Vector)
heatmap(get)
Cutted_Data_Vector = Data_Vector[1000001:end-5000000]
get= Utils.getWaterfall(4e6,Cutted_Data_Vector)
heatmap(get)
filename_cut= "/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run1bis)Preamble/DatFile_Cut/$(List_files[i])"
writeComplexBinary(Cutted_Data_Vector,filename_cut)


List_files = readdir("/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run1bis)Payload/DatFile/")

i=1
filename = "/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run1bis)Payload/DatFile/$(List_files[i])"
Data_Vector= DatBinaryFiles.readComplexBinary(filename)
get= Utils.getWaterfall(4e6,Data_Vector)
heatmap(get)
Cutted_Data_Vector = Data_Vector[1000001:end-5000000]
get= Utils.getWaterfall(4e6,Cutted_Data_Vector)
heatmap(get)
filename_cut= "/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run1bis)Payload/DatFile_Cut/$(List_files[i])"
writeComplexBinary(Cutted_Data_Vector,filename_cut)





Run3="2024-06-04-16h/"

List_files = readdir("/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run3)Preamble/DatFile/")


i=1
filename = "/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run3)Preamble/DatFile/$(List_files[i])"
Data_Vector= DatBinaryFiles.readComplexBinary(filename)
gett= Utils.getWaterfall(4e6,Data_Vector)
heatmap(gett)
Cutted_Data_Vector = Data_Vector[1000001:end-5000000]
get= Utils.getWaterfall(4e6,Cutted_Data_Vector)
heatmap(get)
filename_cut= "/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run3)Preamble/DatFile_Cut/$(List_files[i])"
writeComplexBinary(Cutted_Data_Vector,filename_cut)


List_files = readdir("/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run3)Payload/DatFile/")

i=1
filename = "/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run3)Payload/DatFile/$(List_files[i])"
Data_Vector= DatBinaryFiles.readComplexBinary(filename)
get= Utils.getWaterfall(4e6,Data_Vector)
heatmap(get)
Cutted_Data_Vector = Data_Vector[1000001:end-5000000]
get= Utils.getWaterfall(4e6,Cutted_Data_Vector)
heatmap(get)
filename_cut= "/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run3)Payload/DatFile_Cut/$(List_files[i])"
writeComplexBinary(Cutted_Data_Vector,filename_cut)



Run4="2024-07-05-12h/"


List_files = readdir("/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run4)Preamble/DatFile/")


i=2
filename = "/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run4)Preamble/DatFile/$(List_files[i])"
Data_Vector= DatBinaryFiles.readComplexBinary(filename)
gett= Utils.getWaterfall(4e6,Data_Vector)
heatmap(gett)
Cutted_Data_Vector = Data_Vector[1000001:end-5000000]
get= Utils.getWaterfall(4e6,Cutted_Data_Vector)
heatmap(get)
filename_cut= "/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run4)Preamble/DatFile_Cut/$(List_files[i])"
writeComplexBinary(Cutted_Data_Vector,filename_cut)


List_files = readdir("/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run4)Payload/DatFile/")

i=1
filename = "/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run4)Payload/DatFile/$(List_files[i])"
Data_Vector= DatBinaryFiles.readComplexBinary(filename)
get= Utils.getWaterfall(4e6,Data_Vector)
heatmap(get)
Cutted_Data_Vector = Data_Vector[1000001:end-5000000]
get= Utils.getWaterfall(4e6,Cutted_Data_Vector)
heatmap(get)
filename_cut= "/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run4)Payload/DatFile_Cut/$(List_files[i])"
writeComplexBinary(Cutted_Data_Vector,filename_cut)


Run4bis="2024-07-09-12h/"


List_files = readdir("/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run4bis)Preamble/DatFile/")


i=1
filename = "/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run4bis)Preamble/DatFile/$(List_files[i])"
Data_Vector= DatBinaryFiles.readComplexBinary(filename)
gett= Utils.getWaterfall(4e6,Data_Vector)
heatmap(gett)
Cutted_Data_Vector = Data_Vector[1000001:end-5000000]
gett= Utils.getWaterfall(4e6,Cutted_Data_Vector)
heatmap(gett)
filename_cut= "/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run4bis)Preamble/DatFile_Cut/$(List_files[i])"
writeComplexBinary(Cutted_Data_Vector,filename_cut)


List_files = readdir("/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run4bis)Payload/DatFile/")

i=1
filename = "/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run4bis)Payload/DatFile/$(List_files[i])"
Data_Vector= DatBinaryFiles.readComplexBinary(filename)
gett= Utils.getWaterfall(4e6,Data_Vector)
heatmap(gett)
Cutted_Data_Vector = Data_Vector[1000001:end-5000000]
gett= Utils.getWaterfall(4e6,Cutted_Data_Vector)
heatmap(gett)
filename_cut= "/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run4bis)Payload/DatFile_Cut/$(List_files[i])"
writeComplexBinary(Cutted_Data_Vector,filename_cut)




Run5="2024-07-04-16h/"

List_files = readdir("/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run5)Preamble/DatFile/")


i=1
filename = "/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run5)Preamble/DatFile/$(List_files[i])"
Data_Vector= DatBinaryFiles.readComplexBinary(filename)
gett= Utils.getWaterfall(4e6,Data_Vector)
heatmap(gett)
Cutted_Data_Vector = Data_Vector[1000001:end-5000000]
get= Utils.getWaterfall(4e6,Cutted_Data_Vector)
heatmap(get)
filename_cut= "/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run5)Preamble/DatFile_Cut/$(List_files[i])"
writeComplexBinary(Cutted_Data_Vector,filename_cut)


List_files = readdir("/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run5)Payload/DatFile/")

i=1
filename = "/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run5)Payload/DatFile/$(List_files[i])"
Data_Vector= DatBinaryFiles.readComplexBinary(filename)
get= Utils.getWaterfall(4e6,Data_Vector)
heatmap(get)
Cutted_Data_Vector = Data_Vector[1000001:end-5000000]
get= Utils.getWaterfall(4e6,Cutted_Data_Vector)
heatmap(get)
filename_cut= "/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run5)Payload/DatFile_Cut/$(List_files[i])"
writeComplexBinary(Cutted_Data_Vector,filename_cut)










Run6= "2024-06-06-12h/"



List_files = readdir("/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run4)Preamble/DatFile/")


i=1
filename = "/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run4)Preamble/DatFile/$(List_files[i])"
Data_Vector= DatBinaryFiles.readComplexBinary(filename)
gett= Utils.getWaterfall(4e6,Data_Vector)
heatmap(gett)
Cutted_Data_Vector = Data_Vector[1000001:end-5000000]
get= Utils.getWaterfall(4e6,Cutted_Data_Vector)
heatmap(get)
filename_cut= "/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run4)Preamble/DatFile_Cut/$(List_files[i])"
writeComplexBinary(Cutted_Data_Vector,filename_cut)


List_files = readdir("/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run4)Payload/DatFile/")

i=1
filename = "/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run4)Payload/DatFile/$(List_files[i])"
Data_Vector= DatBinaryFiles.readComplexBinary(filename)
get= Utils.getWaterfall(4e6,Data_Vector)
heatmap(get)
Cutted_Data_Vector = Data_Vector[1000001:end-5000000]
get= Utils.getWaterfall(4e6,Cutted_Data_Vector)
heatmap(get)
filename_cut= "/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run4)Payload/DatFile_Cut/$(List_files[i])"
writeComplexBinary(Cutted_Data_Vector,filename_cut)





Run2= "2024-06-12-17h/"



List_files = readdir("/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run2)Preamble/DatFile/")


i=1
filename = "/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run2)Preamble/DatFile/$(List_files[i])"
Data_Vector= DatBinaryFiles.readComplexBinary(filename)
get= Utils.getWaterfall(4e6,Data_Vector)
heatmap(get)
Cutted_Data_Vector = Data_Vector[1000001:end-5000000]
get= Utils.getWaterfall(4e6,Cutted_Data_Vector)
heatmap(get)
filename_cut= "/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run2)Preamble/DatFile_Cut/$(List_files[i])"
writeComplexBinary(Cutted_Data_Vector,filename_cut)


List_files = readdir("/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run2)Payload/DatFile/")

i=1
filename = "/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run2)Payload/DatFile/$(List_files[i])"
Data_Vector= DatBinaryFiles.readComplexBinary(filename)
get= Utils.getWaterfall(4e6,Data_Vector)
heatmap(get)
Cutted_Data_Vector = Data_Vector[1000001:end-5000000]
get= Utils.getWaterfall(4e6,Cutted_Data_Vector)
heatmap(get)
filename_cut= "/media/redinblack/ANR_RedInBlack/rffExperiment/$(Run2)Payload/DatFile_Cut/$(List_files[i])"
writeComplexBinary(Cutted_Data_Vector,filename_cut)


