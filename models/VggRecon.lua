require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'loadcaffe'

local function createModel(opt)
  local function convBlock(nIn, nOut, upsample)
    local backend = nn
    local module = nn.Sequential()
    if upsample then
      module:add(backend.SpatialFullConvolution(nIn, nOut, 3, 3, 2, 2, 1, 1, 1, 1))
    else
      module:add(backend.SpatialConvolution(nIn, nOut, 3, 3, stride, stride, 1, 1))
    end
    module:add(cudnn.SpatialBatchNormalization(nOut))
    module:add(cudnn.ReLU(true))
    return module
  end

  require 'Debugger'
  local model = nn.Sequential()
    :add(nn.Identity())
    :add(nn.ParallelTable()
      :add(nn.Identity())
      :add(nn.Identity())
      :add(nn.Sequential()
        :add(convBlock(512, 512, true))
        :add(convBlock(512, 512))
        :add(convBlock(512, 512))
        :add(convBlock(512, 512))
      )
    )
    :add(nn.ConcatTable()
      :add(nn.SelectTable(1))
      :add(nn.Sequential()
        :add(nn.NarrowTable(2, 2))
        :add(nn.JoinTable(1, 3))
        :add(convBlock(1024, 256, true))
        :add(convBlock(256, 256))
        :add(convBlock(256, 256))
        :add(convBlock(256, 256))
      )
    )
    :add(nn.JoinTable(1, 3))
    :add(convBlock(512, 128, true))
    :add(convBlock(128, 128))
    :add(convBlock(128, 64, true))
    :add(cudnn.SpatialConvolution(64, 3, 3, 3, 1, 1, 1, 1))

  local function ConvInit(name)
    for k,v in pairs(model:findModules(name)) do
      local n = v.kW*v.kH*v.nOutputPlane
      v.weight:normal(0,math.sqrt(2/n))
      if cudnn.version >= 4000 then
        v.bias = nil
        v.gradBias = nil
      else
        v.bias:zero()
      end
    end
  end
  local function BNInit(name)
    for k,v in pairs(model:findModules(name)) do
      v.weight:fill(1)
      v.bias:zero()
    end
  end

  ConvInit('cudnn.SpatialConvolution')
  BNInit('cudnn.SpatialBatchNormalization')
  model:cuda()

  return model
end

return createModel
