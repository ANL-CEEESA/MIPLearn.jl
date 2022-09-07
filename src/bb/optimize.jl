#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using Printf
using Base.Threads
import Base.Threads: @threads, nthreads, threadid

import ..load_data, ..Hdf5Sample

function solve!(
    mip::MIP;
    time_limit::Float64 = Inf,
    node_limit::Int = typemax(Int),
    gap_limit::Float64 = 1e-4,
    print_interval::Int = 5,
    initial_primal_bound::Float64 = Inf,
    detailed_output::Bool = false,
    branch_rule::VariableBranchingRule = ReliabilityBranching(),
    enable_plunging = true,
)::NodePool
    time_initial = time()
    pool = NodePool(mip=mip)
    pool.primal_bound = initial_primal_bound

    root_node = _create_node(mip)
    if isempty(root_node.fractional_variables)
        println("root relaxation is integer feasible")
        pool.dual_bound = root_node.obj
        pool.primal_bound = root_node.obj
        return pool
    else
        print_progress_header(detailed_output=detailed_output)
    end

    offer(
        pool,
        parent_node=nothing,
        child_nodes=[root_node],
        print_interval=print_interval,
        detailed_output=detailed_output,
    )
    @threads for t = 1:nthreads()
        child_one, child_zero, suggestions = nothing, nothing, Node[]
        while true
            time_elapsed = time() - time_initial
            if enable_plunging && (child_one !== nothing)
                suggestions = Node[child_one, child_zero]
            end
            node = take(
                pool,
                suggestions=suggestions,
                time_remaining=time_limit - time_elapsed,
                node_limit=node_limit,
                gap_limit=gap_limit,
            )
            if node == :END
                break
            elseif node == :WAIT
                sleep(0.1)
                continue
            else
                ids = generate_indices(pool, 2)
                branch_var = find_branching_var(branch_rule, node, pool)

                # Find current variable lower and upper bounds
                offset = findfirst(isequal(branch_var), mip.int_vars)
                var_lb = mip.int_vars_lb[offset]
                var_ub = mip.int_vars_ub[offset]
                for (offset, v) in enumerate(node.branch_vars)
                    if v == branch_var
                        var_lb = max(var_lb, node.branch_lb[offset])
                        var_ub = min(var_ub, node.branch_ub[offset])
                    end
                end

                # Query current fractional value
                offset = findfirst(isequal(branch_var), node.fractional_variables)
                var_value = node.fractional_values[offset]

                child_zero = _create_node(
                    mip,
                    index=ids[2],
                    parent=node,
                    branch_var=branch_var,
                    branch_var_lb=var_lb,
                    branch_var_ub=floor(var_value),
                )
                child_one = _create_node(
                    mip,
                    index=ids[1],
                    parent=node,
                    branch_var=branch_var,
                    branch_var_lb=ceil(var_value),
                    branch_var_ub=var_ub,
                )
                offer(
                    pool,
                    parent_node=node,
                    child_nodes=[child_one, child_zero],
                    time_elapsed=time_elapsed,
                    print_interval=print_interval,
                    detailed_output=detailed_output,
                )
            end
        end
    end
    return pool
end

function _create_node(
    mip;
    index::Int=0,
    parent::Union{Nothing,Node}=nothing,
    branch_var::Union{Nothing,Variable}=nothing,
    branch_var_lb::Union{Nothing,Float64}=nothing,
    branch_var_ub::Union{Nothing,Float64}=nothing
)::Node
    if parent === nothing
        branch_vars = Variable[]
        branch_lb = Float64[]
        branch_ub = Float64[]
        depth = 1
    else
        branch_vars = [parent.branch_vars; branch_var]
        branch_lb = [parent.branch_lb; branch_var_lb]
        branch_ub = [parent.branch_ub; branch_var_ub]
        depth = parent.depth + 1
    end
    set_bounds!(mip, branch_vars, branch_lb, branch_ub)
    status, obj = solve_relaxation!(mip)
    if status == :Optimal
        vals = values(mip, mip.int_vars)
        fractional_indices =
            [j for j in 1:length(mip.int_vars) if 1e-6 < vals[j] - floor(vals[j]) < 1 - 1e-6]
        fractional_values = vals[fractional_indices]
        fractional_variables = mip.int_vars[fractional_indices]
    else
        fractional_variables = Variable[]
        fractional_values = Float64[]
    end
    n_branch = length(branch_vars)
    set_bounds!(mip, branch_vars, zeros(n_branch), ones(n_branch))
    return Node(
        mip,
        index,
        depth,
        obj,
        status,
        branch_vars,
        branch_lb,
        branch_ub,
        fractional_variables,
        fractional_values,
        parent,
    )
end

function solve!(
    optimizer,
    filename::String;
    time_limit::Float64=Inf,
    node_limit::Int=typemax(Int),
    gap_limit::Float64=1e-4,
    print_interval::Int=5,
    detailed_output::Bool=false,
    branch_rule::VariableBranchingRule=ReliabilityBranching()
)::NodePool
    model = read_from_file("$filename.mps.gz")
    mip = init(optimizer)
    load!(mip, model)

    h5 = Hdf5Sample("$filename.h5")
    primal_bound = h5.get_scalar("mip_obj_value")
    nvars = length(h5.get_array("static_var_names"))

    pool = solve!(
        mip;
        initial_primal_bound=primal_bound,
        time_limit,
        node_limit,
        gap_limit,
        print_interval,
        detailed_output,
        branch_rule
    )

    pseudocost_up = [NaN for _ = 1:nvars]
    pseudocost_down = [NaN for _ = 1:nvars]
    priorities = [0.0 for _ in 1:nvars]
    for (var, var_hist) in pool.var_history
        pseudocost_up[var.index] = var_hist.pseudocost_up
        pseudocost_down[var.index] = var_hist.pseudocost_down
        x = mean(var_hist.fractional_values)
        f_up = x - floor(x)
        f_down = ceil(x) - x
        priorities[var.index] = var_hist.pseudocost_up * f_up * var_hist.pseudocost_down * f_down
    end
    h5.put_array("bb_var_pseudocost_up", pseudocost_up)
    h5.put_array("bb_var_pseudocost_down", pseudocost_down)
    h5.put_array("bb_var_priority", priorities)

    return pool
end
