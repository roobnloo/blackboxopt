module BlackBoxOptimization

export IterResult, covariance_matrix_adaptation, cross_entropy_mvn
using LinearAlgebra, Distributions

struct IterResult
    μ::Vector
    Σ::Matrix
    sample::Vector
end

function covariance_matrix_adaptation(f, x, k_max;
    σ = 1.0,
    m = 4 + floor(Int, 3*log(length(x))),
    m_elite = div(m, 2),
    tol = 10e-6)

    μ, n = copy(x), length(x)
    ws = normalize!(vcat(log((m+1)/2) .- log.(1:m_elite), 
                    zeros(m - m_elite)), 1)
    μ_eff = 1 / sum(ws.^2)
    cσ = (μ_eff + 2)/(n + μ_eff + 5)
    dσ = 1 + 2max(0, sqrt((μ_eff - 1)/(n+1)) - 1) + cσ
    cΣ = (4 + μ_eff/n) / (n + 4 + 2μ_eff / n)
    c1 = 2/((n + 1.3)^2 + μ_eff)
    cμ = min(1 - c1, 2 * (μ_eff - 2 + 1/μ_eff)/((n+2)^2 + μ_eff))
    E = n^0.5 * (1 - 1/(4n) + 1/(21 * n^2))
    pσ, pΣ, Σ = zeros(n), zeros(n), Matrix(1.0I, n, n)

    result = Vector{IterResult}(undef, k_max)
    for k in 1 : k_max
        P = MvNormal(μ, σ^2 * Σ)
        xs = [rand(P) for _ in 1:m]
        ys = [f(x) for x in xs]
        is = sortperm(ys) # best to worst

        # selection and mean update
        δs = [(x - μ)/σ for x in xs]
        δw = sum(ws[i] * δs[is[i]] for i in 1:m_elite)
        result[k] = IterResult(μ, σ^2 * Σ, xs[is[1:m_elite]])
        μ += σ * δw

        # step-size control
        C = Σ^-0.5
        pσ = (1-cσ)*pσ + sqrt(cσ * (2-cσ) * μ_eff) * C *δw
        σ *= exp(cσ/dσ * (norm(pσ)/E - 1))

        # covariance adaptation
        hσ = Int(norm(pσ)/sqrt(1-(1-cσ)^(2k)) < (1.4 + 2/(n+1)) * E)
        pΣ = (1-cΣ) * pΣ + hσ * sqrt(cΣ * (2 - cΣ) * μ_eff) * δw
        w0 = [ws[i] ≥ 0 ? ws[i] : n * ws[i] / norm(C * δs[is[i]]) * 2
                for i in 1:m]
        Σ = (1 - c1 - cμ) * Σ +
            c1 * (pΣ * pΣ' + (1-hσ) * cΣ * (2 - cΣ) * Σ) +
            cμ * sum(w0[i] * δs[is[i]] * δs[is[i]]' for i in 1:m)
        Σ = triu(Σ) + triu(Σ, 1)' # enforce symmetry

        if norm(μ - result[k].μ) <= tol
            result = result[1:k]
            break
        end
    end
    return μ, result
end

function cross_entropy_mvn(f, μ, Σ, k_max; m=100, m_elite=10, tol=10e-6)
    result = Vector{IterResult}(undef, k_max)
    P = MvNormal(μ, Σ)
    for k in 1:k_max
        xs = [rand(P) for _ in 1:m]
        ys = [f(x) for x in xs]
        is = sortperm(ys) # best to worst
        elite = xs[is[1:m_elite]]
        result[k] = IterResult(μ, Σ, elite)
        # μ = mean.(elite)
        # Σ = sum((x - μ) * (x - μ)' for x in elite) / m_elite
        P = fit(typeof(P), transpose(reduce(vcat, transpose.(elite))))
        μ = mean(P)
        Σ = cov(P)

        if norm(μ - result[k].μ) <= tol
            result = result[1:k]
            break
        end
    end
    return μ, result
end
end