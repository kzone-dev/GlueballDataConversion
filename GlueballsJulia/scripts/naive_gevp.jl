using Pkg; Pkg.activate("GlueballsJulia")
using HDF5
using ProgressMeter
using LatticeUtils
using LinearAlgebra
BLAS.set_num_threads(1)
@show Threads.nthreads()

function _copy_lattice_parameters(outfile,infile;group="")
    file = h5open(infile)
    entries = filter(!contains(r"(correlation_matrix|singlet_loop|vev)") ,keys(file))
    for entry in entries
        label = joinpath(group,entry)
        h5write(outfile,label,read(file,entry))
    end
end
function eigenvalues_eigenvectors_from_samples_threaded(batch; t0, imag_thresh)
    n_cpus = Threads.nthreads()
    n_batch = size(batch)[3]

    f = x -> eigenvalues_eigenvectors_from_samples(batch[:,:,x,:]; t0, imag_thresh)
    ranges = Iterators.partition(1:n_batch,n_cpus)
    res = fetch.([Threads.@spawn f(i) for i in ranges])

    vals = cat(first.(res)...,dims=2)
    vecs = cat(last.(res)...,dims=3)
    return vals, vecs
end
function gevp_on_resamples(file_in, file_out; n_batch, t0, imag_thresh)
    _copy_lattice_parameters(file_out,file_in)

    fid  = h5open(file_in)
    resamples = fid["correlation_matrix_resample"]

    Nmeas, Nops, Nops, T = size(resamples)

    f = h5open(file_out,"w")
    create_dataset(f, "eigenvalues", Float64, (Nops,Nmeas,T))
    create_dataset(f, "eigenvectors", ComplexF64, (Nops,Nops,Nmeas,T))
    h5write(file_out,"t0",t0)

    @showprogress for n in Iterators.partition(1:Nmeas, n_batch)
        # change data layout of data to match the required call to the gevp function
        batch = permutedims(resamples[n,:,:,:],(3,2,1,4))
        #vals0, vecs0 = eigenvalues_eigenvectors_from_samples(batch; t0, imag_thresh)
        vals , vecs  = eigenvalues_eigenvectors_from_samples_threaded(batch; t0, imag_thresh)
        f["eigenvalues"][:,n,:] = vals
        f["eigenvectors"][:,:,n,:] = vecs
    end
    close(f)
    close(fid)
end

file_in  = "./hdf5/correlation_matrix_g5_resamples.hdf5"
file_out = "./hdf5/gevp_results_g5_resamples_parallel.hdf5"
gevp_on_resamples(file_in, file_out; n_batch = Threads.nthreads()*5, t0 = 1, imag_thresh = +Inf)