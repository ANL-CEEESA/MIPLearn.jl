#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using Printf

function print_progress_header(; detailed_output::Bool)
    @printf(
        "%8s  %9s %9s %13s %13s %13s %9s %8s",
        "time",
        "visited",
        "pending",
        "obj",
        "primal-bound",
        "dual-bound",
        "gap",
        "lp-iter"
    )
    if detailed_output
        @printf(
            " %6s %6s %-24s %6s %6s %6s",
            "node",
            "parent",
            "branch-var",
            "b-val",
            "depth",
            "iinfes"
        )
    end
    println()
    flush(stdout)
end

function print_progress(
    pool::NodePool,
    node::Node;
    time_elapsed::Float64,
    print_interval::Int,
    detailed_output::Bool,
    primal_update::Bool,
)::Nothing
    prefix = primal_update ? "*" : " "
    if (pool.processed % print_interval == 0) || isempty(pool.pending) || primal_update
        @printf(
            "%8.2f %1s%9d %9d %13.6e %13.6e %13.6e %9.2e %8d",
            time_elapsed,
            prefix,
            pool.processed,
            length(pool.processing) + length(pool.pending),
            node.obj * node.mip.sense,
            pool.primal_bound * node.mip.sense,
            pool.dual_bound * node.mip.sense,
            pool.gap,
            pool.mip.lp_iterations,
        )
        if detailed_output
            if isempty(node.branch_variables)
                branch_var_name = "---"
                branch_value = "---"
            else
                branch_var_name = name(node.mip, last(node.branch_variables))
                L = min(24, length(branch_var_name))
                branch_var_name = branch_var_name[1:L]
                branch_value = @sprintf("%.2f", last(node.branch_values))
            end
            @printf(
                " %6d %6s %-24s %6s %6d %6d",
                node.index,
                node.parent === nothing ? "---" : @sprintf("%d", node.parent.index),
                branch_var_name,
                branch_value,
                length(node.branch_variables),
                length(node.fractional_variables)
            )
        end
        println()
        flush(stdout)
    end
end
