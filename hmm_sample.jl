using Distributions
using StatsBase
include("./baum_welch_lib.jl")
using Hmm


# 状態数
state_num = 2
# 出力シンボル(この場合は 0 と 1 )
symbol = [0 1]
# 出力シンボルの数
symbol_num = length(symbol)

# これが本当の値
# 遷移確率行列
A = [0.9 0.1; 0.15 0.85]
# 出力確率行列
B = [0.8 0.2; 0.2 0.8]
# 初期確率
ρ = [0.01 0.99]

# これはパラメータ学習の初期値
# 遷移確率行列
eA = [0.4 0.6; 0.3 0.7]
# 出力確率行列
eB = [0.8 0.2; 0.2 0.8]
# 初期確率
eρ = [1/2 1/2]


srand(1234)

function simulate(nSteps)

  observations = zeros(nSteps)
  states = zeros(nSteps)
  states[1] = sample(symbol, Weights(vec(ρ)))
  observations[1] = sample(symbol, Weights(B[Int(states[1])+1, :]))
  for t in 2:nSteps
    states[t] = sample(symbol, Weights(A[Int(states[t-1])+1, :]))
    observations[t] = sample(symbol, Weights(B[Int(states[t])+1, :]))
  end
  return observations,states
end

o1, s = simulate(500)
o2, s = simulate(20)
o3, s = simulate(9)
o4, s = simulate(3)
obs = [o1, o2, o3, o4]

hmm = Hmm.hmm_initialization(eA, eB, eρ)
Hmm.train(hmm, obs, 1e-4, 100000)

println("Actual parameter ρ")
println(ρ)
println("Estimated parameter ρ")
println(hmm.ρ)
println("Actual parameter A")
println(A)
println("Estimated parameter A")
println(hmm.A)
println("Actual parameter B")
println(B)
println("Estimated parameter B")
println(hmm.B)
