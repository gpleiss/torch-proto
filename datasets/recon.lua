--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  DFI dataset loader
--

local image = require 'image'
local paths = require 'paths'
local t = require 'datasets/transforms'
local ffi = require 'ffi'

local M = {}
local ReconDataset = torch.class('resnet.ReconDataset', M)

function ReconDataset:__init(imageInfo, opt, split)
  self.imageInfo = imageInfo[split]
  self.opt = opt
  self.split = split
  self.dir = paths.concat(opt.datasetDir, split)
  assert(paths.dirp(self.dir), 'directory does not exist: ' .. self.dir)
end

function ReconDataset:get(i)
  local path = self.imageInfo[i]

  local image = self:_loadImage(paths.concat(self.dir, path))

  return {
    input = image,
    target = 1,
    path = path,
  }
end

function ReconDataset:_loadImage(path)
  local ok, input = pcall(function()
    return image.load(path, 3, 'float')
  end)

  if not ok then
    input = torch.FloatTensor(3, 448, 448):zero()
  end

  return input
end

function ReconDataset:size()
  return #self.imageInfo
end

-- Computed from random subset of ImageNet training images
local meanstd = {
  mean = { 0.485, 0.456, 0.406 },
  std = { 0.229, 0.224, 0.225 },
}
local pca = {
  eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
  eigvec = torch.Tensor{
    { -0.5675,  0.7192,  0.4009 },
    { -0.5808, -0.0045, -0.8140 },
    { -0.5836, -0.6948,  0.4203 },
  },
}

function ReconDataset:preprocess()
  if self.split == 'train' then
    return t.Compose{
      t.RandomSizedCrop(448),
      t.ColorJitter({
        brightness = 0.2,
        contrast = 0.2,
        saturation = 0.2,
      }),
      t.Lighting(0.1, pca.eigval, pca.eigvec),
      t.CaffeColorNormalize(meanstd),
      t.HorizontalFlip(0.5),
    }
  elseif self.split == 'val' or self.split == 'test' then
    local Crop = self.opt.tenCrop and t.TenCrop or t.CenterCrop
    return t.Compose{
      t.Scale(512),
      t.CaffeColorNormalize(meanstd),
      Crop(448),
    }
  else
    error('invalid split: ' .. self.split)
  end
end

return M.ReconDataset
