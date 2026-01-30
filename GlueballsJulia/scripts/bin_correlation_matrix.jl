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
function filter_bad_vevs(vev_data, meson_ops_array; config_number = 2507)
    # In the M3 ensemble, after config 2507 there are measurements
    # with bad vev values for the smeared operators. We filter them out here.

    reference_vev_values = dropdims(mean(vev_data[1:config_number,meson_ops_array],dims=1),dims=1)
    reference_vev_values_std = dropdims(std(vev_data[1:config_number,meson_ops_array],dims=1),dims=1)

    Nconfs = size(vev_data,1)
    Nops = size(vev_data,2)
    bad_indices = Bool[false for i in 1:Nconfs]
    
    for iconf in 1:Nconfs
        for (i, jop) in enumerate(meson_ops_array)
            relative_error = abs(vev_data[iconf,jop] - reference_vev_values[i]) / reference_vev_values_std[i]
            if relative_error> 5
                bad_indices[iconf] = true
                println("Excluding configuration ", iconf, " due to operator ", jop,
                        ": vev = ", vev_data[iconf,jop],
                        ", reference vev = ", reference_vev_values[i],
                        ", relative error= ", relative_error)
                break
            end
        end
    end
    return bad_indices
end
function bin_correlation_matrix(file_in,file_out; binsize = 8, batchsize = 50, include_vev = false, filter_vev = false)
    fid   = h5open(file_in)
    Nmeas = read(fid,"Nmeas")
    Nops  = read(fid,"Nops")
    T     = read(fid,"T")

    filtered_Nmeas = 1:Nmeas
    if filter_vev
        bad_indices= filter_bad_vevs(fid["full_vev"], 161:169; config_number = 2507)
        filtered_Nmeas = collect(1:Nmeas)[.!bad_indices]
    end 
    
    iterator    = Iterators.partition(filtered_Nmeas, batchsize*binsize) 
    nbins       = length(Iterators.partition(filtered_Nmeas, binsize))

    corr_binned = zeros(nbins,Nops,Nops,T)
    if include_vev
        vev_binned  = zeros(nbins, Nops)  # only if include_vev = true
    end

    corr_data = read(fid,"full_correlation_matrix")
    vev_data = include_vev ? read(fid,"full_vev") : nothing

    p = Progress(nbins)
    ind = 0
    for n in iterator
        corr_tmp = corr_data[n,:,:,:]
        if include_vev
            vev_tmp = vev_data[n,:]
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
file_out = "/users/nrebelobrito/flavour_singlet_and_glueball_mixing_sp4/data/final_correlation_matrices/M3correlation_matrix_id_bin80.hdf5"
bin_correlation_matrix(file_in,file_out; binsize = 16, batchsize = 100, include_vev = true, filter_vev = true)

file_in  = "/users/nrebelobrito/flavour_singlet_and_glueball_mixing_sp4/data/final_correlation_matrices/M3correlation_matrix_g5.hdf5"
file_out = "/users/nrebelobrito/flavour_singlet_and_glueball_mixing_sp4/data/final_correlation_matrices/M3correlation_matrix_g5_bin80.hdf5"
bin_correlation_matrix(file_in,file_out; binsize = 16, batchsize = 100, include_vev = false)

file_in  = "/users/nrebelobrito/flavour_singlet_and_glueball_mixing_sp4/data/final_correlation_matrices/M4correlation_matrix_id.hdf5"
file_out = "/users/nrebelobrito/flavour_singlet_and_glueball_mixing_sp4/data/final_correlation_matrices/M4correlation_matrix_id_bin80.hdf5"
bin_correlation_matrix(file_in,file_out; binsize = 16, batchsize = 100, include_vev = true)

file_in  = "/users/nrebelobrito/flavour_singlet_and_glueball_mixing_sp4/data/final_correlation_matrices/M4correlation_matrix_g5.hdf5"
file_out = "/users/nrebelobrito/flavour_singlet_and_glueball_mixing_sp4/data/final_correlation_matrices/M4correlation_matrix_g5_bin80.hdf5"
bin_correlation_matrix(file_in,file_out; binsize = 16, batchsize = 100, include_vev = false)
