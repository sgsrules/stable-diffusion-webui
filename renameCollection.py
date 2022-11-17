# import OS module
import glob
import os

# Get the list of all files and directories
path = r'F:\CodeProjects\stable-diffusion-webui\outputs\Flowed\Cybernetica\\'
dir_list = glob.glob(path+"*.*")
#dir_list = os.listdir(path)
filelist = []
#filenames = "Cybel,Fiber,Spanner,Udator,Ofuq,Cyl,Gigabit,Azerty,Ufaz,Adr,Mach,Ratcher,Aro,Eralroid,Experiment,Iczoid,Uval,Bracer,Brobot,Plier,Talus,Scrap,Elix,Egexator,Scrappie,Sark,Andromeda,Avivoid,Sparkle,Greez,Cylinder,Odv,Mechi,Core,Wire,Beta,Uloxator,Ulekoid,Tinker,Buttons,Acmx,Izctron,Hammer,Knave,Rusty,Tera,Golem,Nozzle,Twobit,Spencer,Mecha,Spudnik,Ometron,Ehox,Afkator"
filenames = "Sruzcoun,Baikrez,Qriarzez,Xhutoon,Scerkoq,Zhalzugrix,Zusocog,Cikru,Skudhauk,Aqnaeg,Tujyk,Coloogok,Qriggacor,Iqnoneq,Hodruk,Zhubzoc,Bughaq,Khurogrun,Prexhigas,Bzoktu,Qucheaq,Troqnour,Qrogvan,Sqounqudux,Dhognureed,Szoknaig,Vrotrid,Mhudhughu,Crughykyz,Shogmaux,Shicia,Doldok,Shurqa,Sqomzaac,Tsogde,Vroomkudh,Poktok,Qrarqanoc,Realkolol,Bhoktus,Trikkuc,Zongaagnuc,Somkykox,Nangil,Psescygnos"
filenames = filenames.split(',')
#i = 28
i = 0
for file in dir_list:
    newpath = path + filenames[i] + ".mp4"
    i = i + 1
    os.rename(file, newpath)

