# import OS module
import glob
import os

# Get the list of all files and directories
path = r'F:\CodeProjects\stable-diffusion-webui\outputs\Flowed\Beat Droids\\'
dir_list = glob.glob(path+"*.*")
#dir_list = os.listdir(path)
filelist = []
filenames = "Temjir,Ebium,Surtar,Useus,Vostus,Vurasil,Rhumos,Zania,Moorr,Qhiborh,Rumphion,Briasis,Qhisoi,Qrohrena,Xorla,Myhdona,Iotz,Qutraura,Duanke,Thotana,Vetar,Ikies,Koteus,Qhysnir,Bivmis,Iean,Qheohr,Qhidall,Geas,Zasyn,Vetelia,Xearae,Yneas,Dotrena,Nera,Kruasis,Eyja,Ekta,Enera,Xegtix,Wides,Chimes,Fodagi,Qeher,Osemis,Cedbris,Xadur,Ortar,Ixses,Ynia"
filenames = filenames.split(',')
i = 0
for file in dir_list:
    newpath = path + filenames[i] + ".mp4"
    i = i + 1
    os.rename(file, newpath)

