# 将poscar转成rescu可读的atom.xyz文件
# 提交rescu任务，进行计算
#import numpy as np
import os
from shutil import copyfile

# configure
Pseudopentialpath="Pseudopotential/"
subfile="bash sub.sh"

# program
POSCAR="./POSCAR"
def POTCARsub(infile,subfile) :
    Atomfile="Atom.xyz"
    rescuinput="scf.input"
    with open(infile,"r") as f:
        lines=f.readlines()
    scale=float(lines[1].split()[0])
    a1=list(map(float,lines[2].split()))
    a2=list(map(float,lines[3].split()))
    a3=list(map(float,lines[4].split()))
    ntypes=len(lines[5].split())
    ele=lines[5].split()
    neles=list(map(int,lines[6].split()))
    natoms=0 #np.sum(neles)
    for i in range(ntypes) :
        natoms += neles[i]
        try :
            copyfile(Pseudopentialpath+"%s_DZP.mat" % ele[i], "./"+"%s_DZP.mat" % ele[i] )
        except : 
            print("Unable to copy file. %s" % IOError)
            exit(1)
    print(ntypes,ele,neles,natoms,a1,a2,a3)
    with open(Atomfile,"w") as f:
        f.write(str(natoms)+'\n')
        f.write("AtomType x y z\n")
        iline=8
        for i in range(ntypes) :
            for j in range(neles[i]) :
                f.write(ele[i]+' '+lines[iline])    
                iline += 1
    with open(rescuinput,"w") as f:
        f.write("info.calculationType = 'self-consistent'\n")
        f.write("info.savepath        = './results/scf'\n")
        f.write("%symmetry.spacesymmetry = 0\n")
        f.write("domain.latvec        = [[%lf %lf %lf]; [%lf %lf %lf]; [%lf %lf %lf]]\n" %(a1[0],a1[1],a1[2],a2[0],a2[1],a2[2],a3[0],a3[1],a3[2]))
        f.write("%eigensolver.emptyBand = 16\n")
        f.write("eigensolver.maxit      = 15\n")
        f.write("eigensolver.algo     = 'cfsi'\n")
        f.write("LCAO.status          = 1\n")
        f.write("smearing.sigma       = 0.01\n")
        f.write("units.energy          = 'eV'\n")
        f.write("kpoint.gridn         = [3,3,1]\n")
        f.write("domain.lowres        = 0.4\n")
        f.write("%domain.highres      = 0.3\n")
        f.write("functional.libxc    = 1\n")
        f.write("functional.list      = {'XC_LDA_X','XC_LDA_C_PW'}\n")
        f.write("option.maxSCFiteration = 200\n")
        f.write("mixing.type          = 'density'\n")
        f.write("mixing.method        = 'pulay'\n")
        f.write("mixing.tol           = [1e-05,1e-05]\n")
        f.write("mixing.maxhistory    = 20\n")
        f.write("spin.type            = 'degenerate'\n")
        for i in range(len(ele)) :
            f.write("element(%d).species   = '%s'\n" % (i+1,ele[i]) )
            f.write("element(%d).path      = './%s_DZP.mat'\n" % (i+1,ele[i]) )
        f.write("atom.xyz             = 'Atom.xyz'\n")
        f.write("units.length         = 'Angstrom'\n")
        f.write("gpu.status           = 0\n")
    # os.system(subfile)

POTCARsub(POSCAR,subfile)
