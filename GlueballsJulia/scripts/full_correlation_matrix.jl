using Pkg; Pkg.activate("./GlueballsJulia")
using GlueballsJulia
using ProgressMeter
using HDF5
using BenchmarkTools

function reconstruct_corr(ops)
    Nmeas, Nops, T = size(ops)
    corr = zeros(Nmeas,Nops,Nops,T)
    for t1 in 1:T
        for t2 in 1:T
            Δt = mod(t2-t1,T)
            for op1 in 1:Nops
                for op2 in 1:Nops
                    for n in 1:Nmeas
                        corr[n,op2,op1,Δt+1] += (ops[n,op1,t1]*ops[n,op2,t2] + ops[n,op2,t1]*ops[n,op1,t2])/2
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
Nmeas = read(f_glue,"Nmeas")
Nmeas_mes = length(read(f_ferm,"M3/plaquette"))

corr_meson = f_ferm["M3/correlation_matrix_g5_singlet"][:,:,:,:]
corr_glue  = f_glue["corr"][:,:,:,:]
ops_mesFUN = f_ferm["M3/singlet_loop_g5_FUN"][:,:,:]
ops_mesAS  = f_ferm["M3/singlet_loop_g5_AS"][:,:,:]
ops_glue   = f_glue["ops"][:,:,:]

# Construct full correlation matrices in batches to save rAM
n_batch = 20
@showprogress for n in Iterators.partition(1:Nmeas, n_batch) 
    # Permute indices for better memory acces 
    ops   = permutedims(f["ops"][:,:,n],(3,2,1))  # size is (Nmeas, Nops, T) 
    #corr_reconstruct = reconstruct_corr(ops)
end
=#