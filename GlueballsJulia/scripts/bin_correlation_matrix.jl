using Pkg; Pkg.activate(".")
using HDF5
using Statistics
using ProgressMeter
function _copy_lattice_parameters(outfile,infile;group="")
    file = h5open(infile)
    entries = filter(!contains(r"(correlation_matrix|singlet_loop)") ,keys(file))
    for entry in entries
        label = joinpath(group,entry)
        h5write(outfile,label,read(file,entry))
    end
end
function bin_correlation_marix(file_in,file_out; binsize = 8, batchsize = 50)
    fid   = h5open(file_in)
    Nmeas = read(fid,"Nmeas")
    Nops  = read(fid,"Nops")
    T     = read(fid,"T")

    iterator    = Iterators.partition(1:Nmeas, batchsize*binsize) 
    nbins       = length(Iterators.partition(1:Nmeas, binsize))
    corr_binned = zeros(nbins,Nops,Nops,T)

    p = Progress(nbins)
    ind = 0
    for n in iterator
        corr_tmp = fid["full_correlation_matrix"][n,:,:,:]
        for m in Iterators.partition(1:length(n),binsize)
            ind += 1
            corr_binned[ind,:,:,:] = dropdims(mean(corr_tmp[m,:,:,:],dims=1),dims=1)
            next!(p)
        end
    end

    h5write(file_out,"full_correlation_matrix_binned",corr_binned)
    _copy_lattice_parameters(file_out,file_in;group="")
end

file_in  = "/users/nrebelobrito/Reparsing/output_files/final_matrices/M3correlation_matrix_id.hdf5"
file_out = "/users/nrebelobrito/Reparsing/output_files/final_matrices/M3correlation_matrix_id_bin16.hdf5"
bin_correlation_marix(file_in,file_out; binsize = 16, batchsize = 200)

file_in  = "/users/nrebelobrito/Reparsing/output_files/final_matrices/M3correlation_matrix_g5.hdf5"
file_out = "/users/nrebelobrito/Reparsing/output_files/final_matrices/M3correlation_matrix_g5_bin16.hdf5"
bin_correlation_marix(file_in,file_out; binsize = 16, batchsize = 200)

file_in  = "/users/nrebelobrito/Reparsing/output_files/final_matrices/M4correlation_matrix_id.hdf5"
file_out = "/users/nrebelobrito/Reparsing/output_files/final_matrices/M4correlation_matrix_id_bin16.hdf5"
bin_correlation_marix(file_in,file_out; binsize = 16, batchsize = 200)

file_in  = "/users/nrebelobrito/Reparsing/output_files/final_matrices/M4correlation_matrix_g5.hdf5"
file_out = "/users/nrebelobrito/Reparsing/output_files/final_matrices/M4correlation_matrix_g5_bin16.hdf5"
bin_correlation_marix(file_in,file_out; binsize = 16, batchsize = 200)
