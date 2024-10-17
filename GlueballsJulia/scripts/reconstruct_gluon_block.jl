using Pkg; Pkg.activate("./GlueballsJulia")
using GlueballsJulia
using HDF5

f = h5open("testfile.hdf5","r")["ensemble/0RPpR"]

T = read(f,"NT")
L = read(f,"NX")
Nops  = read(f,"Nop")
Nmeas = read(f,"Nmeas")
corr  = read(f,"corr") # size is ( T/2+1, Nops, Nops, Nmeas)
ops   = read(f,"ops")  # size is (T, Nops, Nmeas) 
vev   = read(f,"vev")  # size is (Nops, Nmeas)
@assert size(corr) == (T÷2+1, Nops, Nops, Nmeas)
@assert size(ops)  == (T, Nops, Nmeas)
@assert size(vev)  == (Nops, Nmeas)

function reconstruct_corr(ops)
    T, Nops, Nmeas = size(ops)
    corr = zeros(T,Nops,Nops,Nmeas)
    for t1 in 1:T
        for t2 in 1:T
            Δt = mod(t2-t1,T)
            for n in 1:Nmeas, op1 in 1:Nops, op2 in 1:Nops
                corr[Δt+1,op1,op2,n] += (ops[t1,op1,n]*ops[t2,op2,n] + ops[t1,op2,n]*ops[t2,op1,n])/2
            end
        end
    end
    return @. corr
end
function test_reconstructiuon(ops,corr0)
    T, Nops, Nmeas = size(ops)
    Tmax  = T÷2+1
    corr1 = reconstruct_corr(ops)
    maxdiff = maximum(abs.(corr0 - corr1[1:Tmax,:,:,:]))
    return maxdiff
end

# test reconstruction first
diff = test_reconstructiuon(ops,corr)
@assert iszero(diff)
corr_reconstruct = reconstruct_corr(ops)