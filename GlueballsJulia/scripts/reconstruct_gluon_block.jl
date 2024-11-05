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
function test_reconstructiuon(ops,corr)
    Nops, Nmeas, T = size(ops)
    Tmax  = T÷2+1
    corr_R = reconstruct_corr(ops,ops)
    maxdiff = maximum(abs.(corr_R[:,:,:,1:Tmax] - corr))
    return maxdiff
end

f = h5open("hdf5/glue_correlators.hdf5","r")["ensemble/0RPpR"]
T = read(f,"NT")
L = read(f,"NX")
Nops  = read(f,"Nop")
Nmeas = read(f,"Nmeas")

# Compare one configuration at a time to use less memory
n_batch = 20
@showprogress for n in Iterators.partition(1:Nmeas, n_batch) 
    # Permute indices for better memory acces 
    corr  = permutedims(f["corr"][:,:,:,n],(4,2,3,1)) # size is (Nmeas Nops, Nops, T/2+1)
    ops   = permutedims(f["ops"][:,:,n],(3,2,1))  # size is (Nmeas, Nops, T) 
    vev   = permutedims(f["vev"][:,n],(2,1))  # size is (Nmeas, Nops)
    corr_reconstruct = reconstruct_corr(ops,ops)
    diff = test_reconstructiuon(ops,corr)
    @assert iszero(diff)
end