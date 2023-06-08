#  MIPLearn: Extensible Framework for Learning-Enhanced Mixed-Integer Optimization
#  Copyright (C) 2020-2023, UChicago Argonne, LLC. All rights reserved.
#  Released under the modified BSD license. See COPYING.md for more details.

using MIPLearn
using JLD2

struct _TestStruct
    n::Int
    q::Vector{Float64}
end

function test_io()
    test_pkl_gz()
    test_jld2()
    test_h5()
end

function test_pkl_gz()
    original = Dict("K1" => 1, "K2" => [0, 1, 2], "K3" => "Hello")
    dirname = tempdir()
    MIPLearn.write_pkl_gz([original], dirname)
    recovered = MIPLearn.read_pkl_gz("$dirname/00000.pkl.gz")
    @test recovered == original
end

function test_h5()
    h5 = H5File(tempname(), "w")
    _test_roundtrip_scalar(h5, "A")
    _test_roundtrip_scalar(h5, true)
    _test_roundtrip_scalar(h5, 1)
    _test_roundtrip_scalar(h5, 1.0)
    @test h5.get_scalar("unknown-key") === nothing
    _test_roundtrip_array(h5, [true, false])
    _test_roundtrip_array(h5, [1, 2, 3])
    _test_roundtrip_array(h5, [1.0, 2.0, 3.0])
    _test_roundtrip_str_array(h5, ["A", "BB", "CCC"])
    @test h5.get_array("unknown-key") === nothing
    h5.close()
end

function test_jld2()
    dirname = mktempdir()
    data = [
        _TestStruct(1, [0.0, 0.0, 0.0]),
        _TestStruct(2, [1.0, 2.0, 3.0]),
        _TestStruct(3, [3.0, 3.0, 3.0]),
    ]
    filenames = write_jld2(data, dirname, prefix="obj")
    @test all(
        filenames .==
        ["$dirname/obj00001.jld2", "$dirname/obj00002.jld2", "$dirname/obj00003.jld2"],
    )
    @assert isfile("$dirname/obj00001.jld2")
    @assert isfile("$dirname/obj00002.jld2")
    @assert isfile("$dirname/obj00003.jld2")
    recovered = read_jld2("$dirname/obj00002.jld2")
    @test recovered.n == 2
    @test all(recovered.q .== [1.0, 2.0, 3.0])
end

function _test_roundtrip_scalar(h5, original)
    h5.put_scalar("key", original)
    recovered = h5.get_scalar("key")
    @test recovered !== nothing
    @test original == recovered
end

function _test_roundtrip_array(h5, original)
    h5.put_array("key", original)
    recovered = h5.get_array("key")
    @test recovered !== nothing
    @test all(original .== recovered)
end

function _test_roundtrip_str_array(h5, original)
    h5.put_array("key", MIPLearn.to_str_array(original))
    recovered = MIPLearn.from_str_array(h5.get_array("key"))
    @test recovered !== nothing
    @test all(original .== recovered)
end
