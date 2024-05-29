#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2024, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using JuMP

global TravelingSalesmanData = PyNULL()
global TravelingSalesmanGenerator = PyNULL()

function __init_problems_tsp__()
    copy!(TravelingSalesmanData, pyimport("miplearn.problems.tsp").TravelingSalesmanData)
    copy!(
        TravelingSalesmanGenerator,
        pyimport("miplearn.problems.tsp").TravelingSalesmanGenerator,
    )
end

function build_tsp_model_jump(data::Any; optimizer)
    nx = pyimport("networkx")

    if data isa String
        data = read_pkl_gz(data)
    end
    model = Model(optimizer)
    edges = [(i, j) for i = 1:data.n_cities for j = (i+1):data.n_cities]
    x = @variable(model, x[edges], Bin)
    @objective(model, Min, sum(x[(i, j)] * data.distances[i, j] for (i, j) in edges))

    # Eq: Must choose two edges adjacent to each node
    @constraint(
        model,
        eq_degree[i in 1:data.n_cities],
        sum(x[(min(i, j), max(i, j))] for j = 1:data.n_cities if i != j) == 2
    )

    function lazy_separate(cb_data)
        x_val = callback_value.(Ref(cb_data), x)
        violations = []
        selected_edges = [e for e in edges if x_val[e] > 0.5]
        graph = nx.Graph()
        graph.add_edges_from(selected_edges)
        for component in nx.connected_components(graph)
            if length(component) < data.n_cities
                cut_edges = [
                    [e[1], e[2]] for
                    e in edges if (e[1] ∈ component && e[2] ∉ component) ||
                    (e[1] ∉ component && e[2] ∈ component)
                ]
                push!(violations, cut_edges)
            end
        end
        return violations
    end

    function lazy_enforce(violations)
        @info "Adding $(length(violations)) subtour elimination eqs..."
        for violation in violations
            constr = @build_constraint(sum(x[(e[1], e[2])] for e in violation) >= 2)
            submit(model, constr)
        end
    end

    return JumpModel(
        model,
        lazy_enforce = lazy_enforce,
        lazy_separate = lazy_separate,
        lp_optimizer = optimizer,
    )
end

export TravelingSalesmanData, TravelingSalesmanGenerator, build_tsp_model_jump
