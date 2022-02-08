include("BlackBoxOptimization.jl")
using .BlackBoxOptimization, Plots, LinearAlgebra

function flower(x; a=1, b=1, c=4)
    return a*norm(x) + b*sin(c*atan(x[2], x[1]))
end

result = covariance_matrix_adaptation(flower, [1,1], 6; m = 10)

x = range(-3, 3, length = 1000)
heat = heatmap(x, x, (w, z) -> flower([w,z]),
                aspect_ratio=:equal,
                legend=:none,
                c=:thermal)

function plot_update(result::CMAIterResult, h::Plots.Plot{Plots.GRBackend})
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

p1 = plot_update(result[2][1], heat)
p2 = plot_update(result[2][2], heat)
p3 = plot_update(result[2][3], heat)
p4 = plot_update(result[2][4], heat)
p5 = plot_update(result[2][5], heat)
p6 = plot_update(result[2][6], heat)
plot(p1, p2, p3, p4, p5, p6, layout = (2, 3), legend = false,
        size = (1200, 800))


