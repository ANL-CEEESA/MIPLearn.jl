global BasicCollector = PyNULL()

function __init_collectors__()
    copy!(BasicCollector, pyimport("miplearn.collectors.basic").BasicCollector)
end

export BasicCollector
