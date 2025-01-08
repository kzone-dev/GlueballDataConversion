using Pkg; Pkg.activate("GlueballsJulia")
using HDF5
using ProgressMeter
using LatticeUtils
function _copy_lattice_parameters(outfile,infile;group="")
    file = h5open(infile)
    entries = filter(!contains(r"(correlation_matrix|singlet_loop|vev)") ,keys(file))
    for entry in entries
        label = joinpath(group,entry)
        h5write(outfile,label,read(file,entry))
    end
end

file_in  = "./hdf5/correlation_matrix_g5_resamples.hdf5"
file_out = "./hdf5/gevp_results_g5_resamples.hdf5"
_copy_lattice_parameters(file_out,file_in)

fid  = h5open(file_in)
resamples = fid["correlation_matrix_resample"]

Nmeas, Nops, Nops, T = size(resamples)
eigvals = zeros(Nmeas, Nops, T)

n_batch = 10
t0 = 1
imag_thresh = 1E-11

f = h5open(file_out,"w")
create_dataset(f, "eigenvalues", Float64, (Nops,Nmeas,T))
create_dataset(f, "eigenvectors", ComplexF64, (Nops,Nops,Nmeas,T))
h5write(file_out,"t0",t0)

@showprogress for n in Iterators.partition(1:Nmeas, n_batch)
    # change data layout of data to match the required call to the gevp function
    batch = permutedims(resamples[n,:,:,:],(3,2,1,4))
    vals, vecs = eigenvalues_eigenvectors_from_samples(batch; t0, imag_thresh)
    f["eigenvalues"][:,n,:] = vals
    f["eigenvectors"][:,:,n,:] = vecs
end
