local multiply_adds = true
local M = {}
local OpCounter = torch.class('OpCounter', M)

function OpCounter:__init(model, opt)
  if torch.type(model) == 'nn.DataParallelTable' then
    model = model:get(1)
  end

  self.model = model
  self.op_count = op_count
  self.op_used = op_used
  self.opt = opt
end

function OpCounter:count(input)
  -- Compute #flops that specified module needs to process an input.
  -- module_handlers table is at the bottom of this file
  local function compute_ops(module, input)
    module_name = torch.type(module)
    handler = OpCounter.module_handlers[module_name]
    assert(handler, string.format("No handler for module %s!", module_name))
    local ops = handler(module, input)
    self.op_count = self.op_count + ops
    table.insert(self.op_used, {name = torch.type(module), ops = ops})
  end

  -- Intercept updateOutput. At each call increment op_count appropriately.
  local function intercept_updateOutput(module)
    module.updateOutput_original = module.updateOutput
    module.updateOutput = function(self, input)
      compute_ops(module, input)
      return module:updateOutput_original(input)
    end
  end

  -- Restore original network behaviour
  local function restore_updateOutput(module)
    assert(module.updateOutput_original,
      "restore_updateOutput should be called after intercept_updateOutput!")
    module.updateOutput = nil
    module.updateOutput_original = nil
  end

  local input
  if string.find(self.opt.dataset, 'cifar') then
    input = torch.rand(1, 3, 32, 32):cuda()
  else
    input = torch.rand(1, 3, 224, 224):cuda()
  end
  self.op_count = 0
  self.op_used = {}
  self.model:apply(intercept_updateOutput)
  self.model:forward(input)
  self.model:apply(restore_updateOutput)
end

function OpCounter:total()
  return self.op_count
end

function OpCounter:byType()
  local per_layer = {}
  for i, info in pairs(self.op_used) do
    local name = info['name']
    local ops = info['ops']
    if ops > 0 then
      if not per_layer[name] then
        per_layer[name] = 0
      end
      per_layer[name] = per_layer[name] + ops
    end
  end

  local result = {}
  for key, val in pairs(per_layer) do
    table.insert(result, {
      layerType = key,
      ops = val,
    })
  end
  table.sort(result, function(a, b) return a.ops > b.ops end)
  return result
end



--------------------------------------------------------------------------------
------------------------------- Module handlers --------------------------------
--------------------------------------------------------------------------------

function OpCounter.ops_nothing(module, input)
  return 0
end

function OpCounter.ops_linear(module, input)
  local batch_size = input:dim() == 2 and input:size(1) or 1
  local weight_ops = module.weight:nElement() * (multiply_adds and 1 or 2)
  local bias_ops = module.bias:nElement()
  local ops_per_sample = weight_ops + bias_ops
  return batch_size * ops_per_sample
end

function OpCounter.ops_logsoftmax(module, input)
  local batch_size = input:dim() == 2 and input:size(1) or 1
  local input_dim = input:dim() == 2 and input:size(2) or input:size(1)
  local expminusapprox_ops = 1 -- around 8 in Torch
  -- +2 for accumulation and substraction in two loops
  local ops_per_elem = expminusapprox_ops + 1 + 1
  local ops_per_sample = input_dim * ops_per_elem
  return batch_size * ops_per_sample
end

-- WARNING: an oversimplified version
function OpCounter.ops_nonlinearity(module, input)
  return input:nElement()
end

function OpCounter.ops_convolution(module, input)
  assert(input:dim() == 4, "OpCounter.ops_convolution supports only batched inputs!")
  assert(input:size(2) == module.nInputPlane, "number of input planes doesn't match!")
  local batch_size = input:size(1)
  local input_planes = input:size(2)
  local input_height = input:size(3)
  local input_width = input:size(4)

  -- ops per output element
  local kernel_ops = module.kH * module.kW * input_planes * (multiply_adds and 1 or 2)
  local bias_ops = 1
  local ops_per_element = kernel_ops + bias_ops

  local output_width = math.floor((input_width + 2 * module.padW - module.kW) / module.dW + 1)
  local output_height = math.floor((input_height + 2 * module.padH - module.kH) / module.dH + 1)

  return batch_size * module.nOutputPlane * output_width * output_height * ops_per_element
end

function OpCounter.ops_pooling(module, input)
  assert(input:dim() == 4, "ops_averagepooling supports only batched inputs!")
  local batch_size = input:size(1)
  local input_planes = input:size(2)
  local input_height = input:size(3)
  local input_width = input:size(4)

  local kernel_ops = module.kH * module.kW

  local output_width = math.floor((input_width + 2 * module.padW - module.kW) / module.dW + 1)
  local output_height = math.floor((input_height + 2 * module.padH - module.kH) / module.dH + 1)

  return batch_size * input_planes * output_width * output_height * kernel_ops
end

function OpCounter.ops_batchnorm(module, input)
  return input:nElement() * (multiply_adds and 1 or 2)
end

OpCounter.module_handlers = {
  -- Containers
  ['nn.Sequential'] = OpCounter.ops_nothing,
  ['nn.Parallel'] = OpCounter.ops_nothing,
  ['nn.Concat'] = OpCounter.ops_nothing,
  ['nn.gModule'] = OpCounter.ops_nothing,
  ['nn.Identity'] = OpCounter.ops_nothing,
  ['nn.DataParallelTable'] = OpCounter.ops_nothing,
  ['nn.Contiguous'] = OpCounter.ops_nothing,
  ['nn.CAddTable'] = OpCounter.ops_nothing,
  ['nn.ConcatTable'] = OpCounter.ops_nothing,
  ['nn.JoinTable'] = OpCounter.ops_nothing,
  ['nn.Padding'] = OpCounter.ops_nothing,

  -- Nonlinearities
  ['nn.ReLU'] = OpCounter.ops_nonlinearity,
  ['nn.LogSoftMax'] = OpCounter.ops_logsoftmax,
  ['cudnn.ReLU'] = OpCounter.ops_nonlinearity,

  -- Basic modules
  ['nn.Linear'] = OpCounter.ops_linear,

  -- Spatial Modules
  ['nn.SpatialConvolution'] = OpCounter.ops_convolution,
  ['nn.SpatialFullConvolution'] = ops_fullconvolution,
  ['nn.SpatialMaxPooling'] = OpCounter.ops_pooling,
  ['nn.SpatialAveragePooling'] = OpCounter.ops_pooling,
  ['nn.SpatialBatchNormalization'] = OpCounter.ops_batchnorm, -- Can be squashed
  ['cudnn.SpatialConvolution'] = OpCounter.ops_convolution,
  ['cudnn.SpatialBatchNormalization'] = OpCounter.ops_batchnorm, -- Can be squashed
  ['cudnn.SpatialMaxPooling'] = OpCounter.ops_pooling,
  ['cudnn.SpatialAveragePooling'] = OpCounter.ops_pooling,

  -- Various modules
  ['nn.View'] = OpCounter.ops_nothing,
  ['nn.Reshape'] = OpCounter.ops_nothing,
  ['nn.Dropout'] = OpCounter.ops_nothing, -- Is turned off in inference
  ['nn.Concat'] = OpCounter.ops_nothing,
  ['nn.MulConstant'] = OpCounter.ops_nothing,
  ['nn.Sigmoid'] = OpCounter.ops_nothing,
  ['nn.GradientReversal'] = OpCounter.ops_nothing,
  ['nn.BatchNormalization'] = OpCounter.ops_nothing,
}
return M.OpCounter
