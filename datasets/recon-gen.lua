--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Script to compute list of ImageNet filenames and classes
--
--  This generates a file gen/imagenet.t7 which contains the list of all
--  ImageNet training and valation images and their classes. This script also
--  works for other datasets arragned with the same layout.
--

local sys = require 'sys'

local M = {}

local function findImages(dir)
  local imagePath = torch.CharTensor()

  ----------------------------------------------------------------------
  -- Options for the GNU and BSD find command
  local extensionList = {'jpg', 'png', 'jpeg', 'JPG', 'PNG', 'JPEG', 'ppm', 'PPM', 'bmp', 'BMP'}
  local findOptions = ' -iname "*.' .. extensionList[1] .. '"'
  for i=2,#extensionList do
    findOptions = findOptions .. ' -o -iname "*.' .. extensionList[i] .. '"'
  end

  -- Find all the images using the find command
  local f = io.popen('find -L ' .. dir .. findOptions)

  local maxLength = -1
  local imagePaths = {}

  -- Generate a list of all the images
  while true do
    local line = f:read('*line')
    if not line then break end

    local filename = paths.basename(line)
    local path = filename

    table.insert(imagePaths, path)

    maxLength = math.max(maxLength, #path + 1)
  end

  f:close()

  -- Convert the generated list to a tensor for faster loading
  local nImages = #imagePaths
  local imagePath = {}
  for i, path in ipairs(imagePaths) do
    table.insert(imagePath, path)
  end

  return imagePath
end

function M.exec(opt, cacheFile)
  -- find the image path names
  local imagePath = torch.CharTensor()  -- path to each image in dataset

  local trainDir = paths.concat(opt.datasetDir, 'train')
  local valDir = paths.concat(opt.datasetDir, 'val')
  local testDir = paths.concat(opt.datasetDir, 'test')
  assert(paths.dirp(trainDir), 'train directory not found: ' .. trainDir)
  assert(paths.dirp(valDir), 'val directory not found: ' .. valDir)
  assert(paths.dirp(testDir), 'test directory not found: ' .. testDir)

  print("=> Generating list of images")

  print(" | finding all train images")
  local trainImagePath = findImages(trainDir)

  print(" | finding all val images")
  local valImagePath = findImages(valDir)

  print(" | finding all test images")
  local testImagePath = findImages(testDir)

  local info = {
    basedir = opt.datasetDir,
    train = trainImagePath,
    val = valImagePath,
    test = testImagePath,
  }

  torch.save(cacheFile, info)
  return info
end

return M
