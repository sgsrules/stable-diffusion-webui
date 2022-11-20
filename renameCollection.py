# import OS module
import glob
import os

# Get the list of all files and directories
path = r'F:\CodeProjects\stable-diffusion-webui\outputs\Flowed\Fractured\\'
dir_list = glob.glob(path+"*.*")
#dir_list = os.listdir(path)
filelist = []
filenames = "Dor,Rhohk,Zuhk,Grulok,Ghan,Amilam,Malachi,Javidan,Aviel,Irik,Gimi,Akilom,Abraham,Anon,Radeon,Drukkok,Drolgrann,Bralgohm,Bragmehk,Grer,Akinen,Yislior,Akividan,Melam,Amviel,Grubbon,Dhom,Rhod,Bhazzlod,Rer,Gakhael,Evkov,Nevidan,Yaven,Ovaviram,Dhozlok,Buc,Drardohk,Dhurdun,Dhadrahn,Hashai,Orukh,Eldam,Nevi,Pindam,Vam,Bruddun,Rhagvud,Gum,Dhunn"
filenames = filenames.split(',')
i = 22
for file in dir_list:
    newpath = path + filenames[i] + ".mp4"
    i = i + 1
    os.rename(file, newpath)

