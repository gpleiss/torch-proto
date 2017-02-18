--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--

local checkpoint = {}

function checkpoint.latest(opt)
  if not opt.resume then
    return nil
  end

  local latestPath = opt.latestFilename
  if not paths.filep(latestPath) then
    return nil
  end

  print('=> Loading checkpoint ' .. latestPath)
  local latest = torch.load(latestPath)
  local optimState = torch.load(latest.optimFilename)
  local logger = torch.load(latest.loggerFilename)
  return latest, optimState, logger
end

function checkpoint.best(opt)
  local bestPath = opt.modelFilename .. '.best'
  if not paths.filep(bestPath) then
    bestPath = opt.modelFilename
    if not paths.filep(bestPath) then
      return nil
    end
  end

  print('=> Loading best model ' .. bestPath)
  local model = torch.load(bestPath)

  local logger
  if paths.filep(opt.loggerFilename) then
    logger = torch.load(opt.loggerFilename)
  else
    logger = {}
  end
  return model, logger
end

function checkpoint.logResults(opt, logger, results)
  for key, val in pairs(results) do
    logger[key] = val
  end
  torch.save(opt.loggerFilename, logger)
end

function checkpoint.save(epoch, model, optimState, logger, isBestModel, opt)
  local function saveModel(m)
    torch.save(opt.modelFilename, m)
    torch.save(opt.optimFilename, optimState)
    torch.save(opt.loggerFilename, logger)
    torch.save(opt.latestFilename, {
      epoch = epoch,
      modelFilename = opt.modelFilename,
      optimFilename = opt.optimFilename,
      loggerFilename = opt.loggerFilename,
    })

    if isBestModel then
      torch.save(opt.modelFilename .. '.best', m)
    end
  end

  -- Remove temporary buffers to reduce checkpoint size
  model:clearState()

  -- Don't save the DataParallelTable for easier loading on other machines
  if torch.type(model) == 'nn.DataParallelTable' then
    saveModel(model:get(1))
  else
    saveModel(model)
  end

  -- Re-use gradInput buffers if the option is set. This is necessary because
  -- of the model:clearState() call clears sharing.
  if opt.shareGradInput then
    local models = require 'models/init'
    models.shareGradInput(model)
  end
end

return checkpoint
