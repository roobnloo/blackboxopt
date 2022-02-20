include("BlackBoxOptimization.jl")
using .BlackBoxOptimization, Plots, LinearAlgebra, Statistics,
        Random, Plots.Measures

function flower(x; a=1, b=1, c=4)
    return a*norm(x) + b*sin(c*atan(x[2], x[1]))
end

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
    return scatter(cont, first.(s), last.(s), legend=:none, c = :white)
end

function flower_test_plot()
    μ0 = [2, 2]

    Random.seed!(102)
    cma_result = covariance_matrix_adaptation(flower, μ0, 12; m = 20)

    Random.seed!(102)
    cemvn_result = cross_entropy_mvn(flower, μ0, Matrix(1.0I, 2, 2), 6; 
                                     m = 20, m_elite = 10)

    x = range(-3, 3, length = 1000)
    heat = heatmap(x, x, (w, z) -> flower([w,z]),
                    aspect_ratio=:equal,
                    legend=:none,
                    c=:haline)

    mean_plot = plot_update(cma_result[2][1], heat)
    update_x = [cma_result[2][2].μ[1], cemvn_result[2][2].μ[1]]
    update_y = [cma_result[2][2].μ[2], cemvn_result[2][2].μ[2]]
    mean_plot = scatter(mean_plot, update_x, update_y, color = ["blue", "red"])
    display(mean_plot)

    cma_plots = [plot_update(cma_result[2][i], heat) for i in 1:12]
    display(plot(cma_plots..., layout = (3, 4), legend = false,
            size = (1200, 900), axis=([], false), margin = 0mm))

    cemvn_plots = [plot_update(cemvn_result[2][i], heat) for i in 1:6]
    display(plot(cemvn_plots..., layout = (2, 3), legend = false,
            size = (900, 600), axis=([], false), margin = 0mm))
end

flower_test_plot()
