#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2023, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using Printf
using Base.Threads
import Base.Threads: @threads, nthreads, threadid

import ..H5File

function solve!(
    mip::MIP;
    time_limit::Float64 = Inf,
    node_limit::Int = typemax(Int),
    gap_limit::Float64 = 1e-4,
    print_interval::Int = 5,
    initial_primal_bound::Float64 = Inf,
    branch_rule::VariableBranchingRule = ReliabilityBranching(),
    enable_plunging = true,
)::NodePool
    time_initial = time()
    pool = NodePool(mip = mip)
    pool.primal_bound = initial_primal_bound

    root_node = _create_node(mip)
    if isempty(root_node.fractional_variables)
        println("root relaxation is integer feasible")
        pool.dual_bound = root_node.obj
        pool.primal_bound = root_node.obj
        return pool
    else
        print_progress_header()
    end

    offer(
        pool,
        parent_node = nothing,
        child_nodes = [root_node],
        print_interval = print_interval,
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
                suggestions = suggestions,
                time_remaining = time_limit - time_elapsed,
                node_limit = node_limit,
                gap_limit = gap_limit,
            )
            if node == :END
                break
            elseif node == :WAIT
                sleep(0.1)
                continue
            else
                # Assert node is feasible
                _set_node_bounds(node)
                status, _ = solve_relaxation!(mip)
                @assert status == :Optimal
                _unset_node_bounds(node)

                # Find branching variable
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
                    index = ids[2],
                    parent = node,
                    branch_var = branch_var,
                    branch_var_lb = var_lb,
                    branch_var_ub = floor(var_value),
                )
                child_one = _create_node(
                    mip,
                    index = ids[1],
                    parent = node,
                    branch_var = branch_var,
                    branch_var_lb = ceil(var_value),
                    branch_var_ub = var_ub,
                )
                offer(
                    pool,
                    parent_node = node,
                    child_nodes = [child_one, child_zero],
                    time_elapsed = time_elapsed,
                    print_interval = print_interval,
                )
            end
        end
    end
    return pool
end

function _create_node(
    mip;
    index::Int = 0,
    parent::Union{Nothing,Node} = nothing,
    branch_var::Union{Nothing,Variable} = nothing,
    branch_var_lb::Union{Nothing,Float64} = nothing,
    branch_var_ub::Union{Nothing,Float64} = nothing,
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
            [j for j = 1:length(mip.int_vars) if 1e-6 < vals[j] - floor(vals[j]) < 1 - 1e-6]
        fractional_values = vals[fractional_indices]
        fractional_variables = mip.int_vars[fractional_indices]
    else
        fractional_variables = Variable[]
        fractional_values = Float64[]
    end
    set_bounds!(mip, mip.int_vars, mip.int_vars_lb, mip.int_vars_ub)
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

function _set_node_bounds(node::Node)
    set_bounds!(node.mip, node.branch_vars, node.branch_lb, node.branch_ub)
end

function _unset_node_bounds(node::Node)
    set_bounds!(node.mip, node.mip.int_vars, node.mip.int_vars_lb, node.mip.int_vars_ub)
end
