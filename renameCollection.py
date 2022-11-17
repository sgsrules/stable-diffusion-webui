# import OS module
import glob
import os

# Get the list of all files and directories
path = r'F:\CodeProjects\stable-diffusion-webui\outputs\Flowed\Beat Droids\\'
dir_list = glob.glob(path+"*.*")
#dir_list = os.listdir(path)
filelist = []
filenames = "Skything,Toptrap,Fangclaw,Jetwar,Growlslide,Weasel,Harness,Wreckage,Flinch,Prodigy,Heavyguard,Retroray,Hydraubite,Growllight,Longguard,Barrage,Sunblast,Snarl,Inferno,Core,Magmastrong,Flamerazor,Chromestream,Dreadhorn,Hailburner,Thunder,Comet,Gadget,Pursuit,Fury,Clifftop,Shockwar,Junkline,Stonedrive,Grimflight,Sky-High,Element,Crossflare,Scratch,Mercy,Deepheap,Meanclaw,Haillock,Twinclutch,Lockkick,Shamble,Torrent,Nightlight,Dread,Mercy,Stormheap,Doublestalker,Steelscraps,Flypunch,Longworks,Fungus,Rush,Prodigy,Residue,Aries,Neutroflow,Hookburner,Blowblaze,Shadowworks,Roadback,Crossflare,Quake,Wallop,Twinkle,Inferno"
filenames = filenames.split(',')
i = 0
for file in dir_list:
    newpath = path + filenames[i] + ".mp4"
    i = i + 1
    os.rename(file, newpath)

