#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using Statistics
using DataStructures
import Base.Threads: threadid

function take(
    pool::NodePool;
    suggestions::Array{Node} = [],
    time_remaining::Float64,
    gap_limit::Float64,
    node_limit::Int,
)::Union{Symbol,Node}
    t = threadid()
    lock(pool.lock) do
        n_processing = length(pool.processing)
        if (
            (pool.gap < gap_limit) ||
            (n_processing + pool.processed >= node_limit) ||
            (time_remaining < 0)
        )
            return :END
        end
        if isempty(pool.pending)
            if isempty(pool.processing)
                return :END
            else
                return :WAIT
            end
        else
            # If one of the suggested nodes is still pending, return it.
            # This is known in the literature as plunging.
            for s in suggestions
                if s in keys(pool.pending)
                    delete!(pool.pending, s)
                    pool.processing[s] = s.obj
                    return s
                end
            end

            # If all suggestions have already been processed
            # or pruned, find another node based on best bound.
            node = dequeue!(pool.pending)
            pool.processing[node] = node.obj
            return node
        end
    end
end

function offer(
    pool::NodePool;
    parent_node::Union{Nothing,Node},
    child_nodes::Vector{Node},
    time_elapsed::Float64 = 0.0,
    print_interval::Int = 100,
    detailed_output::Bool = false,
)::Nothing
    lock(pool.lock) do
        primal_update = false

        # Update node.processing and node.processed
        if parent_node !== nothing
            pool.processed += 1
            delete!(pool.processing, parent_node)
        end

        # Queue child nodes
        for node in child_nodes
            if node.status == :Infeasible
                continue
            end
            if node.obj >= pool.primal_bound - 1e-6
                continue
            end
            if isempty(node.fractional_variables)
                pool.primal_bound = min(pool.primal_bound, node.obj)
                primal_update = true
                continue
            end
            pool.pending[node] = node.obj
        end

        # Update dual bound
        pool.dual_bound = pool.primal_bound
        if !isempty(pool.pending)
            pool.dual_bound = min(pool.dual_bound, peek(pool.pending)[2])
        end
        if !isempty(pool.processing)
            pool.dual_bound = min(pool.dual_bound, peek(pool.processing)[2])
        end

        # Update gap
        if pool.primal_bound == pool.dual_bound
            pool.gap = 0
        else
            pool.gap = abs((pool.primal_bound - pool.dual_bound) / pool.primal_bound)
        end

        if parent_node !== nothing
            # Update branching variable history
            branch_var = child_nodes[1].branch_vars[end]
            offset = findfirst(isequal(branch_var), parent_node.fractional_variables)
            x = parent_node.fractional_values[offset]
            obj_change_up = child_nodes[1].obj - parent_node.obj
            obj_change_down = child_nodes[2].obj - parent_node.obj
            _update_var_history(
                pool = pool,
                var = branch_var,
                x = x,
                obj_change_down = obj_change_down,
                obj_change_up = obj_change_up,
            )
            # Update global history
            pool.history.avg_pseudocost_up =
                mean(vh.pseudocost_up for vh in values(pool.var_history))
            pool.history.avg_pseudocost_down =
                mean(vh.pseudocost_down for vh in values(pool.var_history))
        end

        for node in child_nodes
            print_progress(
                pool,
                node,
                time_elapsed = time_elapsed,
                print_interval = print_interval,
                detailed_output = detailed_output,
                primal_update = isfinite(node.obj) && isempty(node.fractional_variables),
            )
        end
    end
    return
end

function _update_var_history(;
    pool::NodePool,
    var::Variable,
    x::Float64,
    obj_change_down::Float64,
    obj_change_up::Float64,
)::Nothing
    # Create new history entry
    if var ∉ keys(pool.var_history)
        pool.var_history[var] = VariableHistory()
    end
    varhist = pool.var_history[var]

    # Push fractional value
    push!(varhist.fractional_values, x)

    # Push objective value changes
    push!(varhist.obj_change_up, obj_change_up)
    push!(varhist.obj_change_down, obj_change_down)

    # Push objective change ratios
    f_up = x - floor(x)
    f_down = ceil(x) - x
    if isfinite(obj_change_up)
        push!(varhist.obj_ratio_up, obj_change_up / f_up)
    end
    if isfinite(obj_change_down)
        push!(varhist.obj_ratio_down, obj_change_down / f_down)
    end

    # Update variable pseudocosts
    varhist.pseudocost_up = 0.0
    varhist.pseudocost_down = 0.0
    if !isempty(varhist.obj_ratio_up)
        varhist.pseudocost_up = sum(varhist.obj_ratio_up) / length(varhist.obj_ratio_up)
    end
    if !isempty(varhist.obj_ratio_down)
        varhist.pseudocost_down =
            sum(varhist.obj_ratio_down) / length(varhist.obj_ratio_down)
    end
    return
end

function generate_indices(pool::NodePool, n::Int)::Vector{Int}
    lock(pool.lock) do
        result = Int[]
        for i = 1:n
            push!(result, pool.next_index)
            pool.next_index += 1
        end
        return result
    end
end
