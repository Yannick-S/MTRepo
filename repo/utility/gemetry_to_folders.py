import os
from shutil import copyfile, rmtree
from os.path import basename

off_files = os.listdir('gm')
off_files = [f for f in off_files if f[-4:] == ".off"]

if os.path.isdir('gm/raw'):
    rmtree('gm/raw')

os.mkdir('gm/raw')

for file in off_files:
    name = basename(file)[:-4]

    this_path = os.path.join('gm/raw', name)
    print(this_path)
    os.mkdir(this_path)

    os.mkdir(os.path.join( this_path, 'train'))
    os.mkdir(os.path.join( this_path, 'test'))

    copyfile(os.path.join('gm' ,file) , os.path.join(this_path,'train',name + ".off"))
    copyfile(os.path.join('gm' ,file) , os.path.join(this_path,'test',name + ".off"))