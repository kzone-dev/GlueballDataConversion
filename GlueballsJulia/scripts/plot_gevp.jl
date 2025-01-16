using Pkg; Pkg.activate("GlueballsJulia")
using HDF5
using Plots

file_out = "./hdf5/gevp_results_g5_t0_1.hdf5"
f = h5open(file_out)

ev  = read(f,"eigenvalues")
Δev = read(f,"Delta_eigenvalues")

t = 1:6
s = size(ev)[1] - 2
scatter(ev[s,t],yerr=Δev[s,t],yscale=:log10)