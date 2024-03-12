@inline frac(x::Float64) = x - floor(x)

@inline frac2(x::Float64) = ceil(x) - x

function assert_leq(a, b; atol=0.01)
    if !all(a .<= b .+ atol)
        delta = a .- b
        for i in eachindex(delta)
            if delta[i] > atol
                @info "Assertion failed: a[$i] = $(a[i]) <= $(b[i]) = b[$i]"
            end
        end
        error("assert_leq failed")
    end
end

function assert_eq(a, b; atol=1e-4)
    if !all(abs.(a .- b) .<= atol)
        delta = abs.(a .- b)
        for i in eachindex(delta)
            if delta[i] > atol
                @info "Assertion failed: a[$i] = $(a[i]) == $(b[i]) = b[$i]"
            end
        end
        error("assert_eq failed")
    end
end

function assert_cuts_off(cuts::ConstraintSet, x::Vector{Float64}, tol=1e-6)
    for i = 1:length(cuts.lb)
        val = cuts.lhs[i, :]' * x
        if (val <= cuts.ub[i] - tol) && (val >= cuts.lb[i] + tol)
            throw(ErrorException("inequality fails to cut off fractional solution"))
        end
    end
end

function assert_does_not_cut_off(cuts::ConstraintSet, x::Vector{Float64}; tol=1e-6)
    for i = 1:length(cuts.lb)
        val = cuts.lhs[i, :]' * x
        ub = cuts.ub[i]
        lb = cuts.lb[i]
        if (val >= ub) || (val <= lb)
            throw(
                ErrorException(
                    "inequality $i cuts off integer solution ($lb <= $val <= $ub)",
                ),
            )
        end
    end
end
