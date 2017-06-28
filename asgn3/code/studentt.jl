function studentt(d, sigma::Float64, alpha::Float64)
    p = (( (d.^2) ./ (2 * sigma^2) ) .+ 1).^(-alpha)
    return p;
end

function grad_studentt(d, sigma::Float64, alpha::Float64)
    p = ( (-alpha) .* ( d ./ (sigma^2) ) ) .* ( ( (d.^2) ./ (2 * sigma^2) .+ 1) .^ (-alpha-1) )
    return p;
end

function d_alpha_studentt(d, sigma::Float64, alpha::Float64)
  p = -alpha .* (1 .+ (1 / (2 * sigma^2)) .* (d.^2) ).^(-alpha-1)
  return p
end

function d_sigma_studentt(d, sigma::Float64, alpha::Float64)
  p = -alpha .* ((d.^2) ./ (sigma^3)) .* (1 .+ ( (d.^2)./ (2* (sigma^2)) )).^(-alpha-1)
  return p
end

function d_alpha_grad_studentt(d, sigma::Float64, alpha::Float64)
  p = d .* (1/ (sigma^2)) .* ((1 .+ ((d.^2) ./ (2*sigma^2))).^(-alpha-1)) .* (-1 + alpha * log(1 + (1/(2*sigma^2))))
  return p
end

function d_sigma_grad_studentt(d, sigma::Float64, alpha::Float64)
  p = -(alpha/(sigma^3)).*(-2 .* d .* ( (1 .+ ((d.^2) ./ (2*(sigma^2)))).^(-alpha-1)) .- ((d.^3) ./ (sigma^2)) .* (-alpha-1) .* ((1 .+ ((d.^2) ./ (2*(sigma^2)))).^(-alpha-2) ) )
  return p
end
