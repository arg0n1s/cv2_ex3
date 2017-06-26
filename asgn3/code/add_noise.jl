using Distributions

function add_noise(X::Array{Float64,2}, sigma::Float64)
		n_dist = Normal(0,sigma)
		noisy = X + reshape(rand(n_dist, size(X[:])), size(X))
		noisy .+= abs(minimum(noisy))
		noisy .*= (255/ maximum(noisy))
    @assert size(noisy) == size(X)
	return noisy::Array{Float64,2}
end
