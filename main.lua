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
require 'train'
require 'test'
local opts = require 'opts'
local checkpoints = require 'checkpoints'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Load previous checkpoint, if it exists
local checkpoint, optimState, logger = checkpoints.latest(opt)

-- Data loading
local trainLoader, valLoader, testLoader = DataLoader.create(opt)

if not opt.testOnly then
  -- Create model
  local model, criterion = models.setup(opt, checkpoint)
  -- The trainer handles the training loop and evaluation on validation set
  local trainer = Trainer(model, criterion, opt, optimState, logger)
  local validator = Tester(model, opt, logger, 'valid')

  local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
  local bestTop1 = math.huge
  local bestTop5 = math.huge
  for epoch = startEpoch, opt.nEpochs do
    -- Train for a single epoch, run model on validation set
    local trainResults = trainer:train(epoch, trainLoader)
    local valResults = validator:test(epoch, valLoader)

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

-- Testing
local bestModel, logger = checkpoints.best(opt)
local tester = Tester(bestModel, opt, logger, 'test')
local testResults = tester:test(0, testLoader)
checkpoints.logResults(opt, logger, {
  testTop1 = testResults.top1,
  testTop5 = testResults.top5,
})
print(string.format(' * Results top1: %6.3f  top5: %6.3f', testResults.top1, testResults.top5))
