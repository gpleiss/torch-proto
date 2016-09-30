--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The training loop and learning rate schedule
--

local optim = require 'optim'

local M = {
  Runner = require 'runners.runner',
}
local Trainer = torch.class('Trainer', 'Runner', M)

function Trainer:__init(model, criterion, opt, optimState, logger)
  self.model = model
  self.criterion = criterion
  self.optimState = optimState or {
    learningRate = opt.LR,
    learningRateDecay = 0.0,
    momentum = opt.momentum,
    nesterov = true,
    dampening = 0.0,
    weightDecay = opt.weightDecay,
  }
  self.opt = opt
  self.params, self.gradParams = model:getParameters()

  self.logger = logger or {
    iterations = {},
    trainTop1 = {},
    trainTop5 = {},
    trainLoss = {},
    trainTime = {},
    valTop1 = {},
    valTop5 = {},
    valTime = {},
  }
end

function Trainer:train(epoch, dataloader)
  -- Trains the model for a single epoch
  self.optimState.learningRate = self:learningRate(epoch)

  local timer = torch.Timer()
  local dataTimer = torch.Timer()

  local function feval()
    return self.criterion.output, self.gradParams
  end

  local trainSize = dataloader:size()
  local top1Sum, top5Sum, lossSum, timeSum = 0.0, 0.0, 0.0, 0.0
  local N = 0

  print('=> Training epoch # ' .. epoch)
  -- set the batch norm to training mode
  self.model:training()
  for n, sample in dataloader:run() do
    local dataTime = dataTimer:time().real

    -- Copy input and target to the GPU
    self:copyInputs(sample)

    local output = self.model:forward(self.input):float()
    local batchSize = output:size(1)
    local loss = self.criterion:forward(self.model.output, self.target)

    self.model:zeroGradParameters()
    self.criterion:backward(self.model.output, self.target)
    self.model:backward(self.input, self.criterion.gradInput)

    optim.sgd(feval, self.params, self.optimState)

    local top1, top5 = self:computeScore(output, sample.target, 1)
    local time = timer:time().real
    top1Sum = top1Sum + top1*batchSize
    top5Sum = top5Sum + top5*batchSize
    lossSum = lossSum + loss*batchSize
    timeSum = timeSum + time
    N = N + batchSize

    print((' | Epoch: [%d][%d/%d]   Time %.3f  Data %.3f  Err %1.4f  top1 %7.3f  top5 %7.3f'):format(
      epoch, n, trainSize, time, dataTime, loss, top1, top5))

    -- check that the storage didn't get changed do to an unfortunate getParameters call
    assert(self.params:storage() == self.model:parameters()[1]:storage())

    timer:reset()
    dataTimer:reset()
  end

  return {
    top1 = top1Sum / N,
    top5 = top5Sum / N,
    loss = lossSum / N,
    time = timeSum,
  }
end

function Trainer:log(trainResults, valResults)
  table.insert(self.logger.iterations, self.optimState.evalCounter)
  table.insert(self.logger.trainTop1, trainResults.top1)
  table.insert(self.logger.trainTop5, trainResults.top5)
  table.insert(self.logger.trainLoss, trainResults.loss)
  table.insert(self.logger.trainTime, trainResults.time)
  table.insert(self.logger.valTop1, valResults.top1)
  table.insert(self.logger.valTop5, valResults.top5)
  table.insert(self.logger.valTime, valResults.time)
end

function Trainer:learningRate(epoch)
  -- Training schedule
  local decay = 0
  local nEpochs = self.opt.nEpochs
  local epochRatio = (epoch - 1) / nEpochs
  if self.opt.dataset == 'imagenet' then
    decay = math.floor(epochRatio * 3)
  elseif self.opt.dataset == 'cifar10' then
    decay = epochRatio >= 0.75 and 2 or epochRatio >= 0.5 and 1 or 0
  end
  return self.opt.LR * math.pow(0.1, decay)
end

return M.Trainer
