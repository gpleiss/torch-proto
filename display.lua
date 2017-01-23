--
--  Display sample images from the dataset
--

-- Basic setup
local opt, checkpoints, DataLoader = require('setup')()
opt.batchSize = 36

local port = os.getenv('PORT') or 8887
local display = require 'display'
display.configure({port=port})


-- Files for training and testing
local trainLoader, valLoader, testLoader = DataLoader.create(opt)

local _, trainSample = trainLoader:run()()
local _, valSample = valLoader:run()()
local _, testSample = testLoader:run()()
display.image(image.toDisplayTensor({input=trainSample.input, nrow=6}), {title='Train Set'})
display.image(image.toDisplayTensor({input=valSample.input, nrow=6}), {title='Val Set'})
display.image(image.toDisplayTensor({input=testSample.input, nrow=6}), {title='Test Set'})

print('Displaying images at http://0.0.0.0:' .. port)
local size = trainSample.input:size()
print(string.format('Image size: %dx%dx%d', size[2], size[3], size[4]))
