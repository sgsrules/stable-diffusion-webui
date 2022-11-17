# import OS module
import glob
import os

# Get the list of all files and directories
path = r'F:\CodeProjects\stable-diffusion-webui\outputs\Flowed\Beat Droids\\'
dir_list = glob.glob(path+"*.*")
#dir_list = os.listdir(path)
filelist = []
filenames = "Ametrine,Amazonite,Angelite,Apophyllite,Aragonite,Aquamarine,Calcite,Celestite,Amethyst,Cancritine,Epidote,Fluorite,Hematite,Polychronic,Labradorite,Lemurian,Moonstone,Agate,Nuummite,Peridot,Aventurine,Jasper,Quartz,Rhodonite,Ruby,Sapphire,Selenite,Shungite,Alabandite,Sodalite,Turquoise,Carborundum,Ilvaite,Hypersthene,Onyx,Arfvedsonite,Amber,Garnet,Rhodochrosite,Opal"
filenames = filenames.split(',')

i = 0
for file in dir_list:
    newpath = path + filenames[i] + ".mp4"
    i = i + 1
    os.rename(file, newpath)

