include("BlackBoxOptimization.jl")
using .BlackBoxOptimization, Plots, LinearAlgebra, Random

function flower(x; a=1, b=1, c=4)
    return a*norm(x) + b*sin(c*atan(x[2], x[1]))
end

Random.seed!(101)
cma_result = covariance_matrix_adaptation(flower, [1,1], 6; m = 10)
cemvn_result = cross_entropy_mvn(flower, [1,1], Matrix(1.0I, 2, 2), 6, 10, 5)
                            #k_max = 6, m = 10, m_elite=5)

x = range(-3, 3, length = 1000)
heat = heatmap(x, x, (w, z) -> flower([w,z]),
                aspect_ratio=:equal,
                legend=:none,
                c=:thermal)

function plot_update(result::IterResult, h::Plots.Plot{Plots.GRBackend})
    cov1 = result.Σ
    mu1 = result.μ
    s = result.sample
    qf(x, y) = dot([x,y] - mu1, inv(cov1), [x,y] - mu1)

    xs = -3:0.01:3
    ys = -3:0.01:3
    cont = contour(h, xs, ys, qf, levels = 0:0.2:1, clim = (-1, 6),
                    xlim = (-3, 3), ylim = (-3, 3),
                    c = :white, aspect_ratio = :equal,
                    size = (500, 500), legend=:none)
    return scatter(cont, first.(s), last.(s), legend=:none)
end

cma_plots = [plot_update(cma_result[2][i], heat) for i in 1:6]
plot(cma_plots..., layout = (2, 3), legend = false,
        size = (1200, 800))

cemvn_plots = [plot_update(cemvn_result[2][i], heat) for i in 1:6]
plot(cemvn_plots..., layout = (2, 3), legend = false,
        size = (1200, 800))
