#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2023, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using Printf
using Base.Threads
import Base.Threads: @threads, nthreads, threadid

import ..H5File

function collect!(
    optimizer,
    filename::String;
    time_limit::Float64 = Inf,
    node_limit::Int = typemax(Int),
    gap_limit::Float64 = 1e-4,
    print_interval::Int = 5,
    branch_rule::VariableBranchingRule = ReliabilityBranching(collect = true),
    enable_plunging = true,
)::NodePool
    model = read_from_file(filename)
    mip = init(optimizer)
    load!(mip, model)

    h5 = H5File(replace(filename, ".mps.gz" => ".h5"), "r")
    primal_bound = h5.get_scalar("mip_upper_bound")
    if primal_bound === nothing
        primal_bound = h5.get_scalar("mip_obj_value")
    end
    h5.file.close()

    pool = solve!(
        mip;
        initial_primal_bound = primal_bound,
        time_limit,
        node_limit,
        gap_limit,
        print_interval,
        branch_rule,
        enable_plunging,
    )

    h5 = H5File(replace(filename, ".mps.gz" => ".h5"))
    pseudocost_up = [NaN for _ = 1:mip.nvars]
    pseudocost_down = [NaN for _ = 1:mip.nvars]
    priorities = [0.0 for _ = 1:mip.nvars]
    for (var, var_hist) in pool.var_history
        pseudocost_up[var.index] = var_hist.pseudocost_up
        pseudocost_down[var.index] = var_hist.pseudocost_down
        x = mean(var_hist.fractional_values)
        f_up = x - floor(x)
        f_down = ceil(x) - x
        priorities[var.index] =
            var_hist.pseudocost_up * f_up * var_hist.pseudocost_down * f_down
    end
    h5.put_array("bb_var_pseudocost_up", pseudocost_up)
    h5.put_array("bb_var_pseudocost_down", pseudocost_down)
    h5.put_array("bb_var_priority", priorities)
    collect!(branch_rule, h5)
    h5.file.close()

    return pool
end
