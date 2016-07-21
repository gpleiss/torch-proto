--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  ImageNet and CIFAR-10 datasets
--

local Filenames = require 'filenames'
local M = {}

local function isvalid(opt, cachePath)
  local imageInfo = torch.load(cachePath)
  return true
end

function M.create(opt, split)
  local cachePath = Filenames.cacheFile(opt.dataset)
  if not paths.filep(cachePath) or not isvalid(opt, cachePath) then
    paths.mkdir(Filenames.data(opt.dataset))

    local script = paths.dofile(opt.dataset .. '-gen.lua')
    script.exec(opt, cachePath)
  end
  local imageInfo = torch.load(cachePath)

  local Dataset = require('datasets/' .. opt.dataset)
  return Dataset(imageInfo, opt, split)
end

return M
