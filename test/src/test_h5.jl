using MIPLearn

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
