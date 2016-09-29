return function()
  -- Basic setup
  require 'torch'
  require 'paths'
  require 'optim'
  require 'nn'
  require 'nngraph'
  require 'cunn'
  require 'cudnn'
  local opts = require 'opts'
  local opt = opts.parse(arg)

  -- Make torch settings
  torch.setdefaulttensortype('torch.FloatTensor')
  torch.setnumthreads(1)
  torch.manualSeed(opt.manualSeed)
  cutorch.manualSeedAll(opt.manualSeed)

  -- Load custom layers and criteria

  --  Get latest checkpoing and data
  local DataLoader = require 'dataloader'
  local checkpoints = require 'checkpoints'

  return opt, checkpoints, DataLoader
end
