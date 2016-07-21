--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'paths'
require 'optim'
require 'nn'
local DataLoader = require 'dataloader'
local models = require 'models/init'
local Trainer = require 'train'
local opts = require 'opts'
local checkpoints = require 'checkpoints'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Load previous checkpoint, if it exists
local checkpoint, optimState, logger = checkpoints.latest(opt)

-- Create model
local model, criterion = models.setup(opt, checkpoint)

-- Data loading
local trainLoader, valLoader, testLoader = DataLoader.create(opt)

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState, logger)

if not opt.testOnly then
  local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
  local bestTop1 = math.huge
  local bestTop5 = math.huge
  for epoch = startEpoch, opt.nEpochs do
    -- Train for a single epoch
    local trainResults = trainer:train(epoch, trainLoader)

    -- Run model on validation set
    local valResults = trainer:test(epoch, valLoader)

    local bestModel = false
    if valResults.top1 < bestTop1 then
      bestModel = true
      bestTop1 = valResults.top1
      bestTop5 = valResults.top5
      print(' * Best model ', valResults.top1, valResults.top5)
    end

    trainer:log(trainResults, valResults)
    checkpoints.save(epoch, model, trainer.optimState, trainer.logger, bestModel, opt)
  end

  print(string.format(' * Finished top1: %6.3f  top5: %6.3f', bestTop1, bestTop5))
end

local testResults = trainer:test(0, testLoader)
trainer.logger.testTop1 = testResults.top1
trainer.logger.testTop5 = testResults.top5
checkpoints.save(0, model, trainer.optimState, trainer.logger, bestModel, opt)
print(string.format(' * Results top1: %6.3f  top5: %6.3f', testResults.top1, testResults.top5))
