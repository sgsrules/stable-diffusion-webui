# import OS module
import glob
import os

# Get the list of all files and directories
path = r'F:\CodeProjects\stable-diffusion-webui\outputs\Flowed\Beat Droids\\'
dir_list = glob.glob(path+"*.*")
#dir_list = os.listdir(path)
filelist = []
filenames = "Rooth,Zoowk,Skresse,Qeeteth,Gieshroth,Qrecceyeth,Qasaw,Grunew,Reinnuw,Ceshi,Alle,Zoow,Zrook,Gooth,Rheyabei,Reatiyi,Rinebbaw,Rheyia,Ghori,Gishe,Ebben,Zriak,Eisoo,Kebba,Sqirrah,Qaleh,Zesselliek,Sqecosheth,Zhinno,Iecen,Qheeti,Qeaw,Crein,Oli,Scooh,Eirriak,Khuna,Rawa,Ataw,Croot"
filenames = filenames.split(',')
i = 0
for file in dir_list:
    newpath = path + filenames[i] + ".mp4"
    i = i + 1
    os.rename(file, newpath)

