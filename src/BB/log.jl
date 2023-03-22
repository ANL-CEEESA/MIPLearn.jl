#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2023, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using Printf

function print_progress_header()
    @printf(
        "%8s %9s %9s %13s %13s %9s %6s %13s %6s %-24s %9s %9s %6s %6s",
        "time",
        "processed",
        "pending",
        "primal-bound",
        "dual-bound",
        "gap",
        "node",
        "obj",
        "parent",
        "branch-var",
        "branch-lb",
        "branch-ub",
        "depth",
        "iinfes"
    )
    println()
    flush(stdout)
end

function print_progress(
    pool::NodePool,
    node::Node;
    time_elapsed::Float64,
    print_interval::Int,
    primal_update::Bool,
)::Nothing
    if (pool.processed % print_interval == 0) || isempty(pool.pending) || primal_update
        if isempty(node.branch_vars)
            branch_var_name = "---"
            branch_lb = "---"
            branch_ub = "---"
        else
            branch_var_name = name(node.mip, last(node.branch_vars))
            L = min(24, length(branch_var_name))
            branch_var_name = branch_var_name[1:L]
            branch_lb = @sprintf("%9.2f", last(node.branch_lb))
            branch_ub = @sprintf("%9.2f", last(node.branch_ub))
        end
        @printf(
            "%8.2f %9d %9d %13.6e %13.6e %9.2e %6d %13.6e %6s %-24s %9s %9s %6d %6d",
            time_elapsed,
            pool.processed,
            length(pool.processing) + length(pool.pending),
            pool.primal_bound * node.mip.sense,
            pool.dual_bound * node.mip.sense,
            pool.gap,
            node.index,
            node.obj * node.mip.sense,
            node.parent === nothing ? "---" : @sprintf("%d", node.parent.index),
            branch_var_name,
            branch_lb,
            branch_ub,
            length(node.branch_vars),
            length(node.fractional_variables)
        )
        println()
        flush(stdout)
    end
end
