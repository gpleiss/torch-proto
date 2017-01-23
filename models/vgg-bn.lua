--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The ResNet model definition
--

local nn = require 'nn'
require 'cunn'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling

local function createModel(opt)
   local depth = opt.depth
   local iChannels

   -- The basic residual layer block for 18 and 34 layer network, and the
   -- CIFAR networks
   local function basicblock(n, isBig)
      local nInputPlane = iChannels
      iChannels = n

      local s = nn.Sequential()
      if isBig then
        s:add(Convolution(nInputPlane,n,5,5,1,1,2,2))
      else
        s:add(Convolution(nInputPlane,n,3,3,1,1,1,1))
      end
      s:add(cudnn.SpatialBatchNormalization(n))
      s:add(ReLU(true))
      if opt.dropRate > 0 then
        s:add(nn.Dropout(opt.dropRate))
      end

      return s
   end

   -- Creates count residual blocks with specified number of features
   local function layer(block, features, count)
      local s = nn.Sequential()
      for i=1,count do
         s:add(block(features, count == 1 and true))
      end
      s:add(cudnn.SpatialMaxPooling(2, 2, 2, 2))
      return s
   end

   local model = nn.Sequential()

   if opt.dataset == 'cifar10' then
      -- Model type specifies number of layers for CIFAR-10 model
      assert((depth - 3) % 3 == 0, 'depth should be one of 6, 9, 12, 15, 18')
      local n = (depth - 3) / 3
      iChannels = 3
      print(' | VGG-' .. depth .. ' CIFAR-10')

      -- The ResNet CIFAR-10 model
      model:add(layer(basicblock, 64, n))
      model:add(layer(basicblock, 128, n))
      model:add(layer(basicblock, 256, n))
      model:add(nn.View(256 * 4 * 4):setNumInputDims(3))
      model:add(nn.Linear(256 * 4 * 4, 2048))
      model:add(nn.BatchNormalization(2048))
      model:add(cudnn.ReLU(true))
      model:add(nn.Linear(2048, 2048))
      model:add(nn.BatchNormalization(2048))
      model:add(cudnn.ReLU(true))
      model:add(nn.Linear(2048, 10))
   elseif opt.dataset == 'cifar100' then
      -- Model type specifies number of layers for CIFAR-100 model
      assert((depth - 3) % 3 == 0, 'depth should be one of 6, 9, 12, 15, 18')
      local n = (depth - 3) / 3
      iChannels = 3
      print(' | VGG-' .. depth .. ' CIFAR-100')

      model:add(layer(basicblock, 64, n))
      model:add(layer(basicblock, 128, n))
      model:add(layer(basicblock, 256, n))
      model:add(nn.View(256 * 4 * 4):setNumInputDims(3))
      model:add(nn.Linear(256 * 4 * 4, 2048))
      model:add(cudnn.ReLU(true))
      model:add(nn.Dropout(0.5))
      model:add(nn.Linear(2048, 2048))
      model:add(cudnn.ReLU(true))
      model:add(nn.Dropout(0.5))
      model:add(nn.Linear(2048, 100))
   else
      error('invalid dataset: ' .. opt.dataset)
   end

   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('nn.SpatialConvolution')
   BNInit('fbnn.SpatialBatchNormalization')
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end
   model:cuda()

   if opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   model:get(1).gradInput = nil

   print(model)
   return model
end

return createModel
