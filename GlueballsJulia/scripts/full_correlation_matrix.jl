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
    for t1 in 1:T
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

f_glue = h5open("hdf5/glue_correlators.hdf5","r")["ensemble/0RPpR"]
f_ferm = h5open("hdf5/meson_correlators.hdf5","r")
T = read(f_glue,"NT")
L = read(f_glue,"NX")
Nops = read(f_glue,"Nop")
Nmeas_glue = read(f_glue,"Nmeas")
Nmeas_mes = length(read(f_ferm,"M3/plaquette"))
configs = read(f_ferm,"M3/configurations")
Nmeas = min(Nmeas_glue,Nmeas_mes) # NOTE: The last 7 configurations for the glueballs are missing

f = h5open("correlation_matrix.hdf5","w")
create_dataset(f, "full_correlation_matrix", Float64, (Nmeas,169,169,T))

# Construct full correlation matrices in batches to save rAM
n_batch = 20
@showprogress for (i,n) in enumerate(Iterators.partition(1:Nmeas, n_batch)) 
    corr_meson = f_ferm["M3/correlation_matrix_g5_singlet"][:,:,n,:]
    ops_mesFUN = f_ferm["M3/singlet_loop_g5_FUN"][:,n,:]
    ops_mesAS  = f_ferm["M3/singlet_loop_g5_AS"][:,n,:]
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
    
    f["full_correlation_matrix"][n,:,:,:] = full
end
close(f)