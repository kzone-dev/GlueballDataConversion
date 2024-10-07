import os

import numpy as np
import scipy.linalg as la
from numba import jit
from numpy.linalg import inv

import pickle
import sys

import numpy as np

#np.set_printoptions(precision=13)

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

    f0 = "[INFO][FROMDISK] Analysis of \t\t{:d} files."
    print(f0.format(l_fvac))
    vev_type = np.dtype(str(Nop) + "f8")
    raw_tmp = np.empty(bin_size, dtype=vev_type)
    vev = np.empty(Nbin_vev, dtype=vev_type)
    ivevf = 0
    oset = 72
    f0 = "[INFO][FROMDISK] Reading vev \t\t{:d}\t from file {:s}"
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
    f0 = "[INFO][FROMDISK] Reading correlator \t{:d}\t from file {:s}"
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
    f0 = "[INFO][FROMDISK] Reading ops \t{:d}\t from file {:s}"
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
        f0 = "[INFO][FROMDISK] TOT_MEAS = \t{:d}"
        print(f0.format(Nbin * bin_size))
        f0 = "[INFO][FROMDISK] IBIN = \t{:d}"
        print(f0.format(Nbin))
        f0 = "[INFO][FROMDISK] NUMBIN = \t{:d}"
        print(f0.format(Nbin))
    else:
        f0 = "[INFO][FROMDISK] Number of bins read {:d} is less than parameter NUMBIN=[:d}"
        print(f0.format(Nmeas / bin_size, Nbin))

    return vev, corr, ops, Nbin, Tmax, Nop

@jit(nopython=True)
def set_avg(CMAT, CVAC, VNORM, vev, corr, Nbin, Nop, Tmax, irrep):
    """
    This function builds the average over bins
    of the optimal **connected** correlator.
    Performance critical: yes
    Input: the value of corr and vev, the eigenvectors
            vecc, the number of bins, the number of operators
            and the number of timeslices.
    Output: None. the function fills out CMAT and CVAC
    """
    # generate the averaged cvac and cmat

    for i in range(Nop):
        CVAC[Nbin][i] = np.average(vev[:, i]) / ((Tmax - 1.0) * 2.0)

    for i in range(Nop):
        for j in range(Nop):
            for it in range(Tmax):
                sumc = np.average(corr[:, i, j, it])
                CMAT[Nbin][i][j][it] = sumc / ((Tmax - 1.0) * 2.0)

    ###Subtract vev of vacuum if rep is A1++
    if irrep == "0RPpR":
        print("[INFO][REP] A1++ Representation detected, subtracting vev")
        for i in range(Nop):
            for j in range(Nop):
                for it in range(Tmax):
                    CMAT[Nbin][i][j][it] -= CVAC[Nbin][i] * CVAC[Nbin][j]

    for i in range(Nop):
        VNORM[i] = 1 / np.sqrt(CMAT[Nbin][i][i][0])

    for j in range(Nop):
        for i in range(j, Nop):
            for it in range(Tmax):
                sumc = (
                    0.5
                    * (CMAT[Nbin][i][j][it] + CMAT[Nbin][j][i][it])
                    * VNORM[i]
                    * VNORM[j]
                )
                CMAT[Nbin][i][j][it] = sumc
                CMAT[Nbin][j][i][it] = sumc


@jit(nopython=True, parallel=True)
def set_bins(CMAT, CVAC, VNORM, vev, corr, Nbin, Nop, Tmax, irrep):
    """
    This function builds the jacknife bins
    of the optimal **connected** correlator.
    Performance critical: yes
    Input: the value of corr and vev, the eigenvectors
            vecc, the number of bins, the number of operators
            and the number of timeslices.
    Output: None. the function fills out CMAT and CVAC
    """
    for inb in range(Nbin):
        for i in range(Nop):
            sumv = np.sum(vev[:, i])
            CVAC[inb, i] = (sumv - vev[inb, i]) / ((Nbin - 1.0) * (Tmax - 1) * 2.0)

        for i in range(Nop):
            for j in range(Nop):
                for it in range(Tmax):
                    sumc = np.sum(corr[:, i, j, it])
                    CMAT[inb, i, j, it] = (sumc - corr[inb, i, j, it]) / (
                        (Nbin - 1.0) * (Tmax - 1.0) * 2.0
                    )

        # Unfortunate that this check has to be completed for every bin,
        # however it is better to do Nbin checks than having to symmetrize twice per bin

        if irrep == "0RPpR":
            for i in range(Nop):
                for j in range(Nop):
                    for it in range(Tmax):
                        CMAT[inb, i, j, it] -= CVAC[inb, i] * CVAC[inb, j]

        for j in range(Nop):
            for i in range(j, Nop):
                for it in range(Tmax):
                    sumc = (
                        0.5
                        * (CMAT[inb][i][j][it] + CMAT[inb][j][i][it])
                        * VNORM[i]
                        * VNORM[j]
                    )
                    CMAT[inb][i][j][it] = sumc
                    CMAT[inb][j][i][it] = sumc

files_corrs, files_vac, files_ops, irrep = check_files("list_corr", "list_ops", "list_vac")
vev, corr, ops, Nbin, Tmax, Nop = from_disk(files_vac, files_corrs, files_ops, bin_size=1)
print("[INFO][REP] Representation", irrep)
#corr is the raw data for the correlation matrix directly from the fortran code
#ops is the raw data for glueball timeslices

LX4 = (Tmax - 1)*2
Lmax = Tmax

corr_type = np.dtype("(" + str(Nop) + "," + str(Nop) + "," + str(Tmax) + ")f8")
corr3 = np.zeros(Nbin, dtype=corr_type)

#Here I build corr3 from the timeslices, and hopefully it matches corr
print("LX4=",LX4,"Lmax=",Lmax)
for N4 in range(0,LX4):
    for NT in range(0,Tmax):

        N4X = (N4 + NT)% LX4
        #if (N4X > LX4 - 1):
        #    N4X = N4 % LX4

        print("[INFO] CORRELATING TIMESLICES: O(",N4,")O(",N4X,")","NT=",NT)
        for K1 in range(0,Nop):
            for K2 in range(0,Nop):
                for nbin in range(Nbin):
                    corr3[nbin][K1][K2][NT]+=0.5*(ops[nbin][K1][N4]*ops[nbin][K2][N4X])
                    corr3[nbin][K1][K2][NT]+=0.5*(ops[nbin][K2][N4]*ops[nbin][K1][N4X])


        #print(np.average(corr3[:,:,:,it2], axis=0))

print("[COMPARE] RAW DATA AVERAGE (NT = 0) DIRECTLY FROM FORTRAN")
print(np.average(corr[:,[0,1],:,0],axis=2))
print("[COMPARE] RAW DATA AVERAGE (NT = 0) FROM GLUEBALL TIMESLICES")
print(np.average(corr3[:,[0,1],:,0],axis=2))

print("[INFO] NORMALIZING AND SETTING BINS")

#Below we use the set_avg and set_bin routines to normalize and symmetrize the correlator matrix.
#This will produce the average over all configuration as well as the jackknife bins.
CVAC = np.empty(Nbin + 1, dtype=str(Nop) + "f8")
CMAT = np.empty(
    Nbin + 1, dtype="(" + str(Nop) + "," + str(Nop) + "," + str(Tmax) + ")f8"
)
VNORM = np.empty(Nop, dtype="f8")

set_avg(CMAT, CVAC, VNORM, vev, corr, Nbin, Nop, Tmax,irrep)
set_bins(CMAT, CVAC, VNORM, vev, corr, Nbin, Nop, Tmax,irrep)

CVAC2 = np.empty(Nbin + 1, dtype=str(Nop) + "f8")
CMAT2 = np.empty(
    Nbin + 1, dtype="(" + str(Nop) + "," + str(Nop) + "," + str(Tmax) + ")f8"
)
VNORM2 = np.empty(Nop, dtype="f8")

set_avg(CMAT2, CVAC2, VNORM2, vev, corr3, Nbin, Nop, Tmax, irrep)
set_bins(CMAT2, CVAC2, VNORM2, vev, corr3, Nbin, Nop, Tmax, irrep)

print("[COMPARE] CORRELATION MATRIX (NT = 0) DIRECTLY FROM FORTRAN")
print(CMAT2[Nbin,:,:,0])
print("[COMPARE] CORRELATION MATRIX (NT = 0) FROM GLUEBALL TIMESLICES")
print(CMAT2[Nbin,:,:,0])


