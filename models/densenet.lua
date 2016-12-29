require 'nn'
require 'cunn'
require 'cudnn'

local function createModel(opt)
    if (opt.depth - 4 ) % 3 ~= 0 then
      error("Depth must be 3N + 4!")
    end

    if (opt.depth - 4) < opt.mmdLayer then
      error('Invalid mmd layer')
    end

    --#layers in each denseblock
    local N = (opt.depth - 4)/3

    --#channels before entering the first denseblock
    --set it to be comparable with growth rate
    local nChannels = opt.nInitChannels

    local layerNum = 1

    local function addLayer(modelPreMmd, modelPostMmd, nChannels, nOutChannels)
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

      if layerNum <= opt.mmdLayer then
        modelPreMmd:add(concat)
      else
        modelPostMmd:add(concat)
      end
      layerNum = layerNum + 1
    end

    local function addTransition(modelPreMmd, modelPostMmd, nChannels, nOutChannels)
      local container = (layerNum <= opt.mmdLayer) and modelPreMmd or modelPostMmd

      container:add(cudnn.SpatialBatchNormalization(nChannels))
      container:add(cudnn.ReLU(true))
      container:add(cudnn.SpatialConvolution(nChannels, nOutChannels, 1, 1, 1, 1, 0, 0))
      if opt.dropRate > 0 then
        container:add(nn.Dropout(opt.dropRate))
      end
      container:add(cudnn.SpatialAveragePooling(2, 2))
    end

    local modelPreMmd = nn.Sequential()
    local modelPostMmd = nn.Sequential()

    modelPreMmd:add(cudnn.SpatialConvolution(3, nChannels, 3,3, 1,1, 1,1))

    for i=1, N do
      addLayer(modelPreMmd, modelPostMmd, nChannels, opt.growthRate)
      nChannels = nChannels + opt.growthRate
    end
    addTransition(modelPreMmd, modelPostMmd, nChannels, nChannels)

    for i=1, N do
      addLayer(modelPreMmd, modelPostMmd, nChannels, opt.growthRate)
      nChannels = nChannels + opt.growthRate
    end
    addTransition(modelPreMmd, modelPostMmd, nChannels, nChannels)

    for i=1, N do
      addLayer(modelPreMmd, modelPostMmd, nChannels, opt.growthRate)
      nChannels = nChannels + opt.growthRate
    end

    modelPostMmd:add(cudnn.SpatialBatchNormalization(nChannels))
    modelPostMmd:add(cudnn.ReLU(true))
    modelPostMmd:add(cudnn.SpatialAveragePooling(8,8)):add(nn.Reshape(nChannels))
    modelPostMmd:add(nn.Linear(nChannels, opt.nClasses))

    local model1 = - modelPreMmd
    local model2 = model1 - modelPostMmd
    local model = nn.gModule({model1}, {model1, model2}):cuda()

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
