import os
import h5py
import numpy as np
import scipy.linalg as la
from natsort import natsorted
from pathlib import Path

def check_files(fin, fin_ops ,fin_vac, icall=3):
    """
    This function will check that the files list_corr
    and list_vac, containing the correlators in binary
    form, point to the same number of files, and these
    files exist.
    Performance critical: no
    Input: two list files
    Output: arrays containing the path the files
    """

    if not os.path.exists(fin):
        print("List file doesn't exist")
        exit(123)
    with open(fin) as f0:
        f0.seek(0)
        files_corr = f0.readlines()
        num_lines = len(files_corr)

    rep_idx = files_corr[0].find("corr")
    rep_identifier = files_corr[0][rep_idx + 4 : -5]
    print("[INFO][REP] Representation is", rep_identifier)

    if icall == 3:
        if not os.path.exists(fin_vac):
            print("vac list file doesn't exist")
            exit(123)
        with open(fin_vac) as f0:
            f0.seek(0)
            files_vac = f0.readlines()
            num_lines_vac = len(files_vac)

        ### New Block to account for ops file ####
        if not os.path.exists(fin_ops):
            print("ops list file doesn't exist")
            exit(123)
        with open(fin_ops) as f0:
            f0.seek(0)
            files_ops = f0.readlines()
            num_lines_ops = len(files_ops)
        ##########################################

        if num_lines != num_lines_vac:
            print("different number of lines in list files for corr and vac")
            exit(124)

        ### New Block to account for ops file ####
        if num_lines_ops != num_lines_vac:
            print("different number of lines in list files for corr and ops")
            exit(124)

        ##########################################

    return files_corr, files_vac, files_ops, rep_identifier

def read_header(inf):
    """
    This function reads the header part of the
    binary files containing the correlators and return
    the values of the entries
    Performance critical: no
    Input:  path the correlators file
    Output: Values of the entries of the correlator
            files
    """
    h = np.fromfile(inf, dtype=np.float64, count=9)

    NX = int(h[0])
    NY = int(h[1])
    NZ = int(h[2])
    NT = int(h[3])
    Nc = int(h[4])
    Ndata = int(h[5])  # Data is computed in batches, each batch has e.g 20 configs
    bin_size = int(h[6])
    Nshape = int(h[7])
    Nbl = int(h[8])

    # This last one is computed on-the-fly
    Tmax = int(NT / 2 + 1)
    return NX, NY, NZ, NT, Nc, Ndata, bin_size, Nshape, Nbl, Tmax

def from_disk(fvac, fcorr, fops, bin_size):
    """
    This function reads the contents of the binary
    correlator and vev files and stores them in binned
    form in the arrays vev and corr.
    Performance critical: yes
    Input:  array of strings containing the adresses
            of correlators and vev files
    Output: Values of the entries of the correlator
            files
    """
    head = np.empty(10, dtype="f8")
    l_fvac = len(fvac)
    nmeas_vev = 0
    for i in range(l_fvac):
        # removes the last element because it is a '\0'
        ff = fvac[i][:-1]
        head = read_header(ff)
        nmeas_vev += head[5]
    Nop = head[7] * head[8]
    Nbin_vev = nmeas_vev // bin_size

    f0 = "[INFO][ANAGLUEBALLS_TOT] LX1,LX2,LX3,LX4 =  {:d}  {:d}  {:d}  {:d}"
    print(f0.format(head[0], head[1], head[2], head[3]))
    f0 = "[INFO][ANAGLUEBALLS_TOT] BINS FOR ERRORS =    {:d}  MEAS PER BIN =    {:d}"
    print(f0.format(Nbin_vev, bin_size))
    f0 = "[INFO][ANAGLUEBALLS_TOT] Calling FROMDISK"
    print(f0)

    f0 = "[INFO][FROMDISK] Analysis of {:d} files."
    print(f0.format(l_fvac))
    vev_type = np.dtype(str(Nop) + "f8")
    raw_tmp = np.empty(bin_size, dtype=vev_type)
    vev = np.empty(Nbin_vev, dtype=vev_type)
    ivevf = 0
    oset = 72
    f0 = "[INFO][FROMDISK] Reading vev {:d} from file {:s}"
    print(f0.format(ivevf + 1, fvac[ivevf]))

    for inb in range(Nbin_vev):
        for i in range(bin_size):
            ff = fvac[ivevf][:-1]
            raw = np.fromfile(ff, dtype=vev_type, offset=oset, count=1)
            if len(raw) == 0:
                print("[INFO][READ_VAC] End of file.")
                # Switch file and reset offset
                ivevf = ivevf + 1
                ff = fvac[ivevf][:-1]
                print(f0.format(ivevf + 1, ff))
                oset = 72
                raw_tmp[i] = np.fromfile(ff, dtype=vev_type, offset=oset, count=1)
                oset = oset + Nop * 8
            else:
                raw_tmp[i] = raw
                oset = oset + Nop * 8
        vev[inb] = np.sum(raw_tmp, axis=0) / bin_size

    l_fcorr = len(fcorr)
    nmeas_corr = 0
    for i in range(l_fcorr):
        ff = fcorr[i][:-1]
        head = read_header(ff)
        nmeas_corr += head[5]
    Nop = head[7] * head[8]
    Tmax = head[9]
    Nbin_corr = nmeas_corr // bin_size

    corr_type = np.dtype("(" + str(Nop) + "," + str(Nop) + "," + str(Tmax) + ")f8")
    raw_tmp = np.empty(bin_size, dtype=corr_type)
    corr = np.empty(Nbin_corr, dtype=corr_type)
    ivevc = 0
    oset = 72
    f0 = "[INFO][FROMDISK] Reading correlator {:d} from file {:s}"
    print(f0.format(ivevc + 1, fcorr[ivevc][:-1]))
    for inb in range(Nbin_vev):
        for i in range(bin_size):
            ff = fcorr[ivevc][:-1]
            raw = np.fromfile(ff, dtype=corr_type, offset=oset, count=1)
            if len(raw) == 0:
                print("[INFO][READ_CORR] End of file.")
                # Switch file and reset offset
                ivevc = ivevc + 1
                ff = fcorr[ivevc][:-1]
                print(f0.format(ivevc + 1, ff))
                oset = 72
                raw_tmp[i] = np.fromfile(ff, dtype=corr_type, offset=oset, count=1)
                oset = oset + Nop * Nop * Tmax * 8
            else:
                raw_tmp[i] = raw
                oset = oset + Nop * Nop * Tmax * 8
        corr[inb] = np.sum(raw_tmp, axis=0) / bin_size

    ### New Block to account for ops file ####
    l_fops = len(fops)
    nmeas_ops = 0
    for i in range(l_fops):
        ff = fops[i][:-1]
        head = read_header(ff)
        nmeas_ops += head[5]
    Nop = head[7] * head[8]
    Tmax = head[9]
    NT = head[3]
    Nbin_ops = nmeas_ops // bin_size

    ops_type = np.dtype("(" + str(Nop) + "," + str(NT) + ")f8") #Each op has NT, not Tmax
    raw_tmp = np.empty(bin_size, dtype=ops_type)
    ops = np.empty(Nbin_ops, dtype=ops_type)
    ivevc = 0
    oset = 72
    f0 = "[INFO][FROMDISK] Reading ops {:d} from file {:s}"
    print(f0.format(ivevc + 1, fops[ivevc][:-1]))
    for inb in range(Nbin_vev):
        for i in range(bin_size):
            ff = fops[ivevc][:-1]
            raw = np.fromfile(ff, dtype=ops_type, offset=oset, count=1)
            if len(raw) == 0:
                print("[INFO][READ_ops] End of file.")
                # Switch file and reset offset
                ivevc = ivevc + 1
                ff = fops[ivevc][:-1]
                print(f0.format(ivevc + 1, ff))
                oset = 72
                raw_tmp[i] = np.fromfile(ff, dtype=ops_type, offset=oset, count=1)
                oset = oset + Nop * NT * 8 #Offsets must be changed as well
            else:
                raw_tmp[i] = raw
                oset = oset + Nop * NT * 8 #Offsets must be changed as well
        ops[inb] = np.sum(raw_tmp, axis=0) / bin_size
    ##########################################

    if Nbin_vev != Nbin_corr:
        print("problem in number of bins")
    else:
        Nbin = Nbin_vev

    ### New Block to account for ops file ####
    if Nbin_ops != Nbin_corr:
        print("problem in number of bins")
    else:
        Nbin = Nbin_ops
    ##########################################


    Nmeas = Nbin * bin_size
    if Nmeas >= Nbin:
        f0 = "[INFO][FROMDISK] TOT_MEAS = {:d}"
        print(f0.format(Nbin * bin_size))
        f0 = "[INFO][FROMDISK] IBIN = {:d}"
        print(f0.format(Nbin))
        f0 = "[INFO][FROMDISK] NUMBIN = {:d}"
        print(f0.format(Nbin))
    else:
        f0 = "[INFO][FROMDISK] Number of bins read {:d} is less than parameter NUMBIN=[:d}"
        print(f0.format(Nmeas / bin_size, Nbin))

    # FZ: Add more information from the header of the binary files
    NX = int(head[0])
    NY = int(head[1])
    NZ = int(head[2])
    NT = int(head[3])
    Nc = int(head[4])
    Nshape = int(head[7])
    Nblock = int(head[8])
    Nmeas  = nmeas_ops

    return vev, corr, ops, Nbin, Tmax, Nop, NX, NY, NZ, NT, Nc, Nshape, Nblock, Nmeas

def write_listfiles(data_dir,irrep,fn_cor="tmp_cor_list.txt",fn_ops="tmp_ops_list.txt",fn_vac="tmp_vac_list.txt"):
    """
    This function creates a list of the .dat files for the correlators, operators and vevs.
    Performance critical: no
    Input:  String containing a directory, and a string specifying the irrep
    Output: None. Three files are written to disk. The filename can be changed via the optional variables.
    """
    cor_list = [str(x)+"\n" for x in Path(data_dir).glob('**/corr{}.dat'.format(irrep))]
    ops_list = [str(x)+"\n" for x in Path(data_dir).glob('**/PL{}.dat'.format(irrep))]
    vac_list = [str(x)+"\n" for x in Path(data_dir).glob('**/avac0.dat')]
    f_cor = open(fn_cor, "w")
    f_ops = open(fn_ops, "w")
    f_vac = open(fn_vac, "w")
    f_cor.writelines(natsorted(cor_list))
    f_ops.writelines(natsorted(ops_list))
    f_vac.writelines(natsorted(vac_list))
    f_cor.close()
    f_ops.close()
    f_vac.close()

def remove_listfiles(fn_cor, fn_ops ,fn_vac):
    os.remove(fn_cor)
    os.remove(fn_ops)
    os.remove(fn_vac)

def write_irrep_data_to_file(filename, irrep, files_corrs, files_vac, files_ops, vev, corr, ops, Nbin, Tmax, Nop, NX, NY, NZ, NT, Nc, Nshape, Nblock, Nmeas, mode='a'):
    ensemble = "ensemble"+"/"+irrep
    f = h5py.File(filename, mode)
    f.create_dataset(ensemble+"/"+"files_corrs",data=files_corrs)
    f.create_dataset(ensemble+"/"+"files_vac"  ,data=files_vac)
    f.create_dataset(ensemble+"/"+"files_ops"  ,data=files_ops)
    f.create_dataset(ensemble+"/"+"vev"  ,data=vev)
    f.create_dataset(ensemble+"/"+"corr" ,data=corr)
    f.create_dataset(ensemble+"/"+"ops"  ,data=ops)
    f.create_dataset(ensemble+"/"+"Nbin" ,data=Nbin)
    f.create_dataset(ensemble+"/"+"Tmax" ,data=Tmax)
    f.create_dataset(ensemble+"/"+"Nop"  ,data=Nop)
    f.create_dataset(ensemble+"/"+"NX"  ,data=NX)
    f.create_dataset(ensemble+"/"+"NY"  ,data=NY)
    f.create_dataset(ensemble+"/"+"NZ"  ,data=NZ)
    f.create_dataset(ensemble+"/"+"NT"  ,data=NT)
    f.create_dataset(ensemble+"/"+"Nc"  ,data=Nc)
    f.create_dataset(ensemble+"/"+"Nshape" ,data=Nshape)
    f.create_dataset(ensemble+"/"+"Nblock" ,data=Nblock)
    f.create_dataset(ensemble+"/"+"Nmeas"  ,data=Nmeas)

irreps  = ["0RPmR","0RPpR"]

data_dir = "/home/fabian/Documents/Physics/Data/DataCSD/Glueballs"
data_dir = "/home/fabian/Dokumente/Physics/Data/DataCSD/GlueballsNt64"

for ir in irreps:
    write_listfiles(data_dir,ir)
    files_corrs, files_vac, files_ops, irrep = check_files("tmp_cor_list.txt", "tmp_ops_list.txt", "tmp_vac_list.txt");
    vev, corr, ops, Nbin, Tmax, Nop, NX, NY, NZ, NT, Nc, Nshape, Nblock, Nmeas = from_disk(files_vac, files_corrs, files_ops, bin_size=1);
    write_irrep_data_to_file("hdf5/glue_correlators_Nt64_mf0-0.70.hdf5", irrep, files_corrs, files_vac, files_ops, vev, corr, ops, Nbin, Tmax, Nop, NX, NY, NZ, NT, Nc, Nshape, Nblock, Nmeas, mode='a')
    remove_listfiles("tmp_cor_list.txt", "tmp_ops_list.txt", "tmp_vac_list.txt")