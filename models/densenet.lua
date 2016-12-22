require 'nn'
require 'cunn'
require 'cudnn'

local function createModel(opt)
    if (opt.depth - 4 ) % 3 ~= 0 then
      error("Depth must be 3N + 4!")
    end

    --#layers in each denseblock
    local N = (opt.depth - 4)/3

    --#channels before entering the first denseblock
    --set it to be comparable with growth rate
    local nChannels = opt.nInitChannels

    local function addLayer(model, nChannels, nOutChannels)
      concat = nn.Concat(2)
      concat:add(nn.Identity())

      convFactory = nn.Sequential()
      convFactory:add(cudnn.SpatialBatchNormalization(nChannels))
      convFactory:add(cudnn.ReLU(true))
      convFactory:add(cudnn.SpatialConvolution(nChannels, nOutChannels, 3, 3, 1, 1, 1,1))
      if opt.dropRate > 0 then
        convFactory:add(nn.Dropout(opt.dropRate))
      end
      concat:add(convFactory)
      model:add(concat)
    end

    local function addTransition(model, nChannels, nOutChannels)
      model:add(cudnn.SpatialBatchNormalization(nChannels))
      model:add(cudnn.ReLU(true))
      model:add(cudnn.SpatialConvolution(nChannels, nOutChannels, 1, 1, 1, 1, 0, 0))
      if opt.dropRate > 0 then
        model:add(nn.Dropout(opt.dropRate))
      end
      model:add(cudnn.SpatialAveragePooling(2, 2))
    end

    model = nn.Sequential()

    model:add(cudnn.SpatialConvolution(3, nChannels, 3,3, 1,1, 1,1))

    for i=1, N do
      addLayer(model, nChannels, opt.growthRate)
      nChannels = nChannels + opt.growthRate
    end
    addTransition(model, nChannels, nChannels)

    for i=1, N do
      addLayer(model, nChannels, opt.growthRate)
      nChannels = nChannels + opt.growthRate
    end
    addTransition(model, nChannels, nChannels)

    for i=1, N do
      addLayer(model, nChannels, opt.growthRate)
      nChannels = nChannels + opt.growthRate
    end

    model:add(cudnn.SpatialBatchNormalization(nChannels))
    model:add(cudnn.ReLU(true))
    model:add(cudnn.SpatialAveragePooling(8,8)):add(nn.Reshape(nChannels))
    model:add(nn.Linear(nChannels, opt.nClasses))

    --Initialization following ResNet
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
    ConvInit('nn.SpatialConvolution')
    BNInit('fbnn.SpatialBatchNormalization')
    BNInit('cudnn.SpatialBatchNormalization')
    BNInit('nn.SpatialBatchNormalization')
    for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
    end
    model:cuda()

    return model
end

return createModel
