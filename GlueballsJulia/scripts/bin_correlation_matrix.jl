using Pkg; Pkg.resolve(); Pkg.instantiate(); Pkg.update(); Pkg.precompile(); Pkg.activate(".")
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
function bin_correlation_marix(file_in,file_out; binsize = 8, batchsize = 50, include_vev = false)
    fid   = h5open(file_in)
    Nmeas = read(fid,"Nmeas")
    Nops  = read(fid,"Nops")
    T     = read(fid,"T")

    iterator    = Iterators.partition(1:Nmeas, batchsize*binsize) 
    nbins       = length(Iterators.partition(1:Nmeas, binsize))
    corr_binned = zeros(nbins,Nops,Nops,T)
    if include_vev
        vev_binned  = zeros(nbins, Nops)  # only if include_vev = true
    end

    println("Size of loaded correlation matrix chunk: ", size(fid["full_correlation_matrix"]))
    if include_vev
        println("Size of loaded vev chunk: ", size(fid["full_vev"]))
    end

    p = Progress(nbins)
    ind = 0
    for n in iterator
        corr_tmp = fid["full_correlation_matrix"][n,:,:,:]
        if include_vev
            vev_tmp = fid["full_vev"][n,:]
            println("binning a chunk of vev size: ", size(vev_tmp))
        end
        for m in Iterators.partition(1:length(n),binsize)
            ind += 1
            corr_binned[ind,:,:,:] = dropdims(mean(corr_tmp[m,:,:,:],dims=1),dims=1)
            if include_vev
                vev_binned[ind,:] = dropdims(mean(vev_tmp[m,:],dims=1),dims=1)
            end
            next!(p)
        end
    end

    h5write(file_out,"full_correlation_matrix_binned",corr_binned)
    if include_vev
        h5write(file_out,"full_vev_binned",vev_binned)
    end
    _copy_lattice_parameters(file_out,file_in;group="")
end

file_in  = "/users/nrebelobrito/flavour_singlet_and_glueball_mixing_sp4/data/final_correlation_matrices/M3correlation_matrix_id.hdf5"
file_out = "/users/nrebelobrito/flavour_singlet_and_glueball_mixing_sp4/data/final_correlation_matrices/M3correlation_matrix_id_bin16.hdf5"
bin_correlation_marix(file_in,file_out; binsize = 16, batchsize = 50, include_vev = true)

file_in  = "/users/nrebelobrito/flavour_singlet_and_glueball_mixing_sp4/data/final_correlation_matrices/M3correlation_matrix_g5.hdf5"
file_out = "/users/nrebelobrito/flavour_singlet_and_glueball_mixing_sp4/data/final_correlation_matrices/M3correlation_matrix_g5_bin16.hdf5"
bin_correlation_marix(file_in,file_out; binsize = 16, batchsize = 50, include_vev = false)

file_in  = "/users/nrebelobrito/flavour_singlet_and_glueball_mixing_sp4/data/final_correlation_matrices/M4correlation_matrix_id.hdf5"
file_out = "/users/nrebelobrito/flavour_singlet_and_glueball_mixing_sp4/data/final_correlation_matrices/M4correlation_matrix_id_bin16.hdf5"
bin_correlation_marix(file_in,file_out; binsize = 16, batchsize = 50, include_vev = true)

file_in  = "/users/nrebelobrito/flavour_singlet_and_glueball_mixing_sp4/data/final_correlation_matrices/M4correlation_matrix_g5.hdf5"
file_out = "/users/nrebelobrito/flavour_singlet_and_glueball_mixing_sp4/data/final_correlation_matrices/M4correlation_matrix_g5_bin16.hdf5"
bin_correlation_marix(file_in,file_out; binsize = 16, batchsize = 50, include_vev = false)