using PyPlot
using Optim
using Images

include("psnr.jl")
include("problem1.jl")
include("mrf_posterior.jl")
include("studentt.jl")

function grad_loss(gt::Array{Float64,2}, X::Array{Float64,2})
	fac = -20/(sum((gt-X).^2)*log(10))
	G = (gt-X)*fac
	return G::Array{Float64,2}
end

function prediction(X::Array{Float64,2}, Y::Array{Float64,2}, sigma_noise::Float64,
					sigma::Float64, alpha::Float64)
					F = X + grad_mrf_denoise_nlposterior(X, Y, sigma_noise, sigma, alpha)
					alpha_helper = (x,a,s)->d_alpha_grad_studentt(x,s,a).*(studentt(x,s,a).^-1)-grad_studentt(x,s,a).*(studentt(x,s,a).^-2).*d_alpha_studentt(x,s,a)
					sigma_helper = (x,a,s)->d_sigma_grad_studentt(x,s,a).*(studentt(x,s,a).^-1)-grad_studentt(x,s,a).*(studentt(x,s,a).^-2).*d_sigma_studentt(x,s,a)
					dsigma = zeros(Float64, size(X))
					dalpha = zeros(Float64, size(X))
			    for i in 1:size(X,1)
			        for j in 1:size(X,2)
			            da = 0
									ds = 0
			            if j < size(X,2)
											da += alpha_helper(X[i,j]-X[i,j+1],sigma, alpha)
											ds += sigma_helper(X[i,j]-X[i,j+1],sigma, alpha)
			            end
			            if j > 1
											da += alpha_helper(X[i,j-1]-X[i,j],sigma, alpha)
											ds += sigma_helper(X[i,j-1]-X[i,j],sigma, alpha)
			            end
			            if i < size(X,1)
											da += alpha_helper(X[i,j]-X[i+1,j],sigma, alpha)
											ds += sigma_helper(X[i,j]-X[i+1,j],sigma, alpha)
			            end
			            if i > 1
											da += alpha_helper(X[i-1,j]-X[i,j],sigma, alpha)
											ds += sigma_helper(X[i-1,j]-X[i,j],sigma, alpha)
			            end
			            dalpha[i,j]=-da
									dsigma[i,j]=-ds
			        end
			    end

	return F::Array{Float64,2}, dsigma::Array{Float64,2}, dalpha::Array{Float64,2}
end

function learning_objective(gt::Array{Float64,2}, X::Array{Float64,2}, Y::Array{Float64,2},
							sigma_noise::Float64, sigma::Float64, alpha::Float64)
	f, dalpha, dsigma = prediction(X, Y, sigma_noise, sigma, alpha)
	J = -psnr(gt, f)
	g = zeros(Float64, 2)
	g[2] = sum(grad_loss(gt, f) .* -dalpha)
	g[1] = sum(grad_loss(gt, f) .* -dsigma)

	return J::Float64, g::Array{Float64,1}
end

function f(gt::Array{Float64,2}, X::Array{Float64,2}, Y::Array{Float64,2},
							sigma_noise::Float64, theta::Array{Float64, 1})
		J,_ = learning_objective(gt, X, Y, sigma_noise, theta[1], theta[2])
    return J
end

function g(gt::Array{Float64,2}, X::Array{Float64,2}, Y::Array{Float64,2},
							sigma_noise::Float64, theta::Array{Float64, 1}, storage::Array{Float64, 1})
		_,grad = learning_objective(gt, X, Y, sigma_noise, theta[1], theta[2])
    storage[1:end] = grad
end

function find_optimal_params(gt::Array{Float64,2}, X::Array{Float64,2}, Y::Array{Float64,2}, sigma_noise::Float64, theta0)
	res = optimize(x->f(gt,X,Y,sigma_noise,x),(x,storage)->g(gt,X,Y,sigma_noise,x,storage), theta0, LBFGS(), Optim.Options(show_trace=true, iterations=50))
  display(res)
  minTheta = Optim.minimizer(res)
	return minTheta::Array{Float64,1}
end


# Problem 3: Image Denoising with Loss-based Training

function load_images()
  img = 255.*channelview(float64.(Gray.(load("..//data//la.png"))))
  img_noisy = 255.*channelview(float64.(Gray.(load("..//data//la-noisy.png"))))
  return img::Array{Float64, 2}, img_noisy::Array{Float64, 2}
end

function problem3()
	sigma_noise = 15.0
	theta0 = [10.0, 1.0]
	img, img_noisy = load_images()
	PyPlot.figure()
  imshow(img, "gray")
  PyPlot.figure()
  imshow(img_noisy, "gray")
	theta = find_optimal_params(img, img_noisy, img_noisy, sigma_noise, theta0)
	display(theta)
end
