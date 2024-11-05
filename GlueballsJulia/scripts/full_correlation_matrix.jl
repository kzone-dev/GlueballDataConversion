using Pkg; Pkg.activate("./GlueballsJulia")
using GlueballsJulia
using ProgressMeter
using HDF5
using BenchmarkTools

function reconstruct_corr(ops1,ops2)
    Nmeas1, Nops1, T1 = size(ops1)
    Nmeas2, Nops2, T2 = size(ops2)
    @assert T1 == T2
    @assert Nmeas1 == Nmeas1
    Nmeas = Nmeas1 
    T = T1 

    corr = zeros(Nmeas,Nops1,Nops2,T)
    @inbounds for t1 in 1:T
        for t2 in 1:T
            Δt = mod(t2-t1,T)
            for op1 in 1:Nops1
                for op2 in 1:Nops2
                    for n in 1:Nmeas
                        corr[n,op1,op2,Δt+1] += (ops1[n,op1,t1]*ops2[n,op2,t2] + ops2[n,op2,t1]*ops1[n,op1,t2])/2
                    end
                end
            end
        end
    end
    return corr
end


id_glue = "0RPmR"
id_ferm = "g5" 

id_glue = "0RPpR"
id_ferm = "id" 

f_glue = h5open("hdf5/glue_correlators.hdf5","r")["ensemble/$id_glue"]
f_ferm = h5open("hdf5/meson_correlators.hdf5","r")["M3"]
T = read(f_glue,"NT")
L = read(f_glue,"NX")
Nops_glue = read(f_glue,"Nop")
Nops_mes  = length(read(f_ferm,"Wuppertal_levels_FUN")) + length(read(f_ferm,"Wuppertal_levels_AS"))
Nmeas_glue = read(f_glue,"Nmeas")
Nmeas_mes = length(read(f_ferm,"plaquette"))
configs = read(f_ferm,"configurations")
Nmeas = min(Nmeas_glue,Nmeas_mes) # NOTE: The last 7 configurations for the glueballs are missing
Nops = Nops_glue + Nops_mes

# Create a hdf5 dataset for the full correlation matrix without ever loading it into memory
# Only, later we will write to the file in batches.
f = h5open("hdf5/correlation_matrix_$id_ferm.hdf5","w")
create_dataset(f, "full_correlation_matrix_$id_ferm", Float64, (Nmeas,Nops,Nops,T))

# Construct full correlation matrices in batches to save rAM
n_batch = 100
@showprogress for (i,n) in enumerate(Iterators.partition(1:Nmeas, n_batch)) 
    corr_meson = f_ferm["correlation_matrix_$(id_ferm)_singlet"][:,:,n,:]
    ops_mesFUN = f_ferm["singlet_loop_$(id_ferm)_FUN"][:,n,:]
    ops_mesAS  = f_ferm["singlet_loop_$(id_ferm)_AS"][:,n,:]
    ops_glue   = f_glue["ops"][:,:,n]
    ops_mes    = cat(ops_mesFUN, ops_mesAS,dims=1) # NOTE: The meson operators are ordered as (FUN, AS) in ascending order
    
    # Change memory layout for better access patterns
    ops_mesFUN = permutedims(ops_mesFUN,(2,1,3))
    ops_mesAS = permutedims(ops_mesAS,(2,1,3))
    ops_glue = permutedims(ops_glue,(3,2,1))
    corr_mes = permutedims(corr_meson,(3,1,2,4))
    ops_mes = permutedims(ops_mes,(2,1,3))

    # Create block-diagonal gluon matrix and meson-gluon cross corelator
    corr_glue  = reconstruct_corr(ops_glue,ops_glue)
    corr_cross = reconstruct_corr(ops_glue,ops_mes)
    corr_cross_transpose = permutedims(corr_cross,(1,3,2,4))

    # assemble full correlation matrix
    coloumn1 = cat(corr_glue,corr_cross,dims=3)
    coloumn2 = cat(corr_cross_transpose,corr_mes,dims=3)
    full = cat(coloumn1,coloumn2,dims=2)
    
    f["full_correlation_matrix_$(id_ferm)"][n,:,:,:] = full
end
f["T"] = T
f["L"] = L
f["Nmeas"] = Nmeas
f["Nops"] = Nops
close(f)