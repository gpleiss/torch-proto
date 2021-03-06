-- Basic setup
local matio = require 'matio'
local opt, checkpoints, DataLoader = require('setup')()

local trainLoader, valLoader, testLoader = DataLoader.create(opt)
local model = checkpoints.best(opt)
model:evaluate()

local indices = {}
for i = 1, model:size() do
  if torch.typename(model:get(i)) == 'nn.Concat' then
    print(i, model:get(i):get(2):get(3))
    table.insert(indices, i)
  end
end

function test()
  local maxNumBatches = math.ceil(3840 / opt.batchSize)

  local inputs = {}
  local labels = {}
  local split = {}

  for n, sample in trainLoader:run() do
    if n > maxNumBatches then
      break
    end

    table.insert(inputs, sample.input:clone())
    table.insert(labels, sample.target:clone())
    table.insert(split, torch.Tensor(sample.target:size()):fill(1))
  end

  for n, sample in valLoader:run() do
    if n > maxNumBatches then
      break
    end

    table.insert(inputs, sample.input:clone())
    table.insert(labels, sample.target:clone())
    table.insert(split, torch.Tensor(sample.target:size()):fill(0))
  end

  local lastIndex = 1
  print(inputs)
  for filenameIndex, index in ipairs(indices) do
    local features = {}
    local input = torch.CudaTensor()
    while lastIndex <= index do
      for j = 1, #inputs do
        inputs[j] = model:get(lastIndex):forward(inputs[j]:cuda()):float()
        if lastIndex == index then
          table.insert(features, model:get(lastIndex):get(2):get(3).output:float())
        end
      end
      model:get(lastIndex):clearState()
      print(features)
      print(inputs)
      lastIndex = lastIndex + 1
    end

    local outputsFilename = paths.concat(opt.save, string.format("outputs-%03d.mat", filenameIndex))
    local outputs = {
      features = torch.cat(features, 1):float(),
      labels = torch.cat(labels, 1),
      split = torch.cat(split, 1),
    }
    print(outputsFilename, outputs)
    matio.save(outputsFilename, outputs)
  end
end

test()
