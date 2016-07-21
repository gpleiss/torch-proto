--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--

local Filenames = require 'filenames'

local checkpoint = {}

function checkpoint.latest(opt)
  if not opt.resume and not opt.testOnly then
    return nil
  end

  local latestPath = Filenames.latest()
  if not paths.filep(latestPath) then
    return nil
  end

  print('=> Loading checkpoint ' .. latestPath)
  local latest = torch.load(latestPath)
  local optimState = torch.load(latest.optimFile)
  local logger = torch.load(latest.loggerFile)
  return latest, optimState, logger
end

function checkpoint.save(epoch, model, optimState, logger, isBestModel, opt)
  local function saveModel(m)
    local modelFile = Filenames.model()
    local optimFile = Filenames.optimState()
    local loggerFile = Filenames.logger()

    torch.save(modelFile, m)
    torch.save(optimFile, optimState)
    torch.save(loggerFile, logger)
    torch.save(Filenames.latest(), {
      epoch = epoch,
      modelFile = modelFile,
      optimFile = optimFile,
      loggerFile = loggerFile,
    })

    if isBestModel then
      torch.save(Filenames.model() .. '.best', m)
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
