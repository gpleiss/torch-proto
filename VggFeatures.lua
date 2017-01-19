require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'loadcaffe'

local VggFeatures, Parent = torch.class('nn.VggFeatures', 'nn.Module')

function VggFeatures:__init(opt)
  local origModel = loadcaffe.load(paths.concat(opt.lossModel, 'VGG_ILSVRC_19_layers_deploy_fullconv.prototxt'), paths.concat(opt.lossModel, 'vgg_normalised.caffemodel'), 'cudnn')

  local input = nn.Identity()
  local conv3_1 = nn.Sequential()
  for i = 1, 12 do
    conv3_1:add(origModel:get(i))
  end
  local conv4_1 = nn.Sequential()
  for i = 13, 21 do
    conv4_1:add(origModel:get(i))
  end
  local conv5_1 = nn.Sequential()
  for i = 22, 30 do
    conv5_1:add(origModel:get(i))
  end

  local model = nn.Sequential()
    :add(conv3_1)
    :add(nn.ConcatTable()
      :add(nn.Identity())
      :add(conv4_1)
    )
    :add(nn.FlattenTable())
    :add(nn.ConcatTable()
      :add(nn.Identity())
      :add(nn.Sequential()
        :add(nn.SelectTable(2))
        :add(conv5_1)
      )
    )
    :add(nn.FlattenTable())

  self.model = model
  self.output = self.model.output
  self.gradInput = self.model.gradInput

  if opt.nGPU > 1 then
    local gpus = torch.range(1, opt.nGPU):totable()
    local fastest, benchmark = cudnn.fastest, cudnn.benchmark

    local dpt = nn.DataParallelTable(1, true, true)
      :add(self.model, gpus)
      :threads(function()
        local cudnn = require 'cudnn'
        cudnn.fastest, cudnn.benchmark = fastest, benchmark
      end)

    self.model = dpt:cuda()
  end
end

function VggFeatures:updateOutput(input)
  self.output = self.model:updateOutput(input)
  return self.output
end

function VggFeatures:backward(input, gradOutput)
  local res = self.model:updateGradInput(input, gradOutput)
  self.gradInput = self.model.gradInput
  return self.gradInput
end

function VggFeatures:updateGradInput(input, gradOutput)
  local res = self.model:updateGradInput(input, gradOutput)
  self.gradInput = self.model.gradInput
  return self.gradInput
end

function VggFeatures:accGradParameters(input, gradOutput)
end

function VggFeatures:accUpdateGradParameters(input, gradOutput)
end


return createModel
