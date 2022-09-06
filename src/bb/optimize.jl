#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using Printf
using Base.Threads
import Base.Threads: @threads, nthreads, threadid

function solve!(
    mip::MIP;
    time_limit::Float64 = Inf,
    node_limit::Int = typemax(Int),
    gap_limit::Float64 = 1e-4,
    print_interval::Int = 5,
    initial_primal_bound::Float64 = Inf,
    detailed_output::Bool = false,
    branch_rule::VariableBranchingRule = HybridBranching(),
    enable_plunging = true,
)::NodePool
    time_initial = time()
    pool = NodePool(mip = mip)
    pool.primal_bound = initial_primal_bound
    print_progress_header(detailed_output = detailed_output)

    root_node = _create_node(mip)
    if isempty(root_node.fractional_variables)
        println("root relaxation is integer feasible")
        pool.dual_bound = root_node.obj
        pool.primal_bound = root_node.obj
        return pool
    end

    offer(pool, parent_node = nothing, child_nodes = [root_node])
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
                ids = generate_indices(pool, 2)
                branch_variable = find_branching_var(branch_rule, node, pool)
                child_zero = _create_node(
                    mip,
                    index = ids[1],
                    parent = node,
                    branch_variable = branch_variable,
                    branch_value = 0.0,
                )
                child_one = _create_node(
                    mip,
                    index = ids[2],
                    parent = node,
                    branch_variable = branch_variable,
                    branch_value = 1.0,
                )
                offer(
                    pool,
                    parent_node = node,
                    child_nodes = [child_one, child_zero],
                    time_elapsed = time_elapsed,
                    print_interval = print_interval,
                    detailed_output = detailed_output,
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
    branch_variable::Union{Nothing,Variable} = nothing,
    branch_value::Union{Nothing,Float64} = nothing,
)::Node
    if parent === nothing
        branch_variables = Variable[]
        branch_values = Float64[]
        depth = 1
    else
        branch_variables = [parent.branch_variables; branch_variable]
        branch_values = [parent.branch_values; branch_value]
        depth = parent.depth + 1
    end
    set_bounds!(mip, branch_variables, branch_values, branch_values)
    status, obj = solve_relaxation!(mip)
    if status == :Optimal
        vals = values(mip, mip.binary_variables)
        fractional_indices =
            [j for j in 1:length(mip.binary_variables) if 1e-6 < vals[j] < 1 - 1e-6]
        fractional_values = vals[fractional_indices]
        fractional_variables = mip.binary_variables[fractional_indices]
    else
        fractional_variables = Variable[]
        fractional_values = Float64[]
    end
    n_branch = length(branch_variables)
    set_bounds!(mip, branch_variables, zeros(n_branch), ones(n_branch))
    return Node(
        mip,
        index,
        depth,
        obj,
        status,
        branch_variables,
        branch_values,
        fractional_variables,
        fractional_values,
        parent,
    )
end
