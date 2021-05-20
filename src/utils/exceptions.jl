#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2021, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using PyCall
traceback = pyimport("traceback")


macro python_call(expr)
    quote
        try
            return $(esc(expr))
        catch e
            if isa(e, PyCall.PyError)
                printstyled("Uncaught Python exception:\n", bold=true, color=:red)
                traceback.print_exception(e.T, e.val, e.traceback)
            end
            rethrow()
        end
    end
end
