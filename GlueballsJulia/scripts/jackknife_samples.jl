using Pkg; Pkg.activate("./GlueballsJulia")
using GlueballsJulia
using ProgressMeter
using HDF5
using BenchmarkTools
using Statistics

function _copy_lattice_parameters(outfile,infile,ensemble;group="")
    file = h5open(infile)[ensemble]
    entries = filter(!contains(r"(correlation_matrix|singlet_loop)") ,keys(file))
    for entry in entries
        label = joinpath(group,entry)
        h5write(outfile,label,read(file,entry))
    end
end
function resample_full_correlator(file_in, file_out; n_batch = 100)
    _copy_lattice_parameters(file_out,file_in,"")
    f1 = h5open(file_in,"r")
    f2 = h5open(file_out,"w")

    Nmeas = read(f1,"Nmeas")
    Nops  = read(f1,"Nops")
    T     = read(f1,"T")
    corr  = f1["full_correlation_matrix"]
    
    # Construct full correlation matrices in batches to save RAM
    resample_corr = zeros(n_batch,Nops,Nops,T)
    resample_sum  = zeros(Nops,Nops,T)

    # Create a hdf5 dataset for the full correlation matrix without ever loading it into memory
    # Only, later we will write to the file in batches.
    create_dataset(f2, "correlation_matrix_resample", Float64, (Nmeas,Nops,Nops,T))

    # sum over all configurations, then subtract the one configuration each to get a jackknife resample
    @showprogress for n in Iterators.partition(1:Nmeas, n_batch)
        resample_sum .= resample_sum .+ dropdims(sum(corr[n,:,:,:],dims=1),dims=1) 
    end

    # remove one configuration each and save the resulting resample to the hdf5 file
    # only load the correlator data in batches to save time on IO
    @showprogress for n in Iterators.partition(1:Nmeas, n_batch)
        batch = corr[n,:,:,:]
        @inbounds for t in 1:T, ind3 in 1:Nops, ind2 in 1:Nops, ind1 in eachindex(n)
            resample_corr[ind1,ind2,ind3,t] = resample_sum[ind2,ind3,t] - batch[ind1,ind2,ind3,t]
        end
        f2["correlation_matrix_resample"][n,:,:,:] = resample_corr[eachindex(n),:,:,:]
    end
    close(f1)
    close(f2)
end

file_in  ="hdf5/correlation_matrix_g5.hdf5"
file_out ="hdf5/correlation_matrix_g5_resamples.hdf5"    
resample_full_correlator(file_in, file_out)

file_in  ="hdf5/correlation_matrix_id.hdf5"
file_out ="hdf5/correlation_matrix_id_resamples.hdf5"    
resample_full_correlator(file_in, file_out)