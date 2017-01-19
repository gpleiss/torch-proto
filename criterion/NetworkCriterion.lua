require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'loadcaffe'

require 'VggFeatures'
local NetworkCriterion, Parent = torch.class('nn.NetworkCriterion', 'nn.Criterion')

function NetworkCriterion:__init(vggFeatures)
  Parent.__init(self)

  self.vggFeatures = vggFeatures
  self.gradInput = self.vggFeatures.gradInput

  -- Create MSE loss
  self.criterion3_1 = nn.MSECriterion()
  self.criterion4_1 = nn.MSECriterion()
  self.criterion5_1 = nn.MSECriterion()
  --self.criterion3_1.sizeAverage = false
  --self.criterion4_1.sizeAverage = false
  --self.criterion5_1.sizeAverage = false
end

function NetworkCriterion:type(t)
  self.vggFeatures = self.vggFeatures:type(t)
  self.criterion3_1 = self.criterion3_1:type(t)
  self.criterion4_1 = self.criterion4_1:type(t)
  self.criterion5_1 = self.criterion5_1:type(t)
  return Parent.type(self, t)
end

function NetworkCriterion:updateOutput(input, target)
  local modelOutput = self.vggFeatures:updateOutput(input)
  local output3_1 = self.criterion3_1:updateOutput(modelOutput[1], target[1])
  local output4_1 = self.criterion4_1:updateOutput(modelOutput[2], target[2])
  local output5_1 = self.criterion5_1:updateOutput(modelOutput[3], target[3])
  self.output = output3_1 + output4_1 + output5_1
  return self.output
end

function NetworkCriterion:updateGradInput(input, target)
  self.criterion3_1:updateGradInput(self.vggFeatures.output[1], target[1])
  self.criterion4_1:updateGradInput(self.vggFeatures.output[2], target[2])
  self.criterion5_1:updateGradInput(self.vggFeatures.output[3], target[3])
  self.gradInput = self.vggFeatures:updateGradInput(input, {self.criterion3_1.gradInput, self.criterion4_1.gradInput, self.criterion5_1.gradInput})
  return self.gradInput
end
