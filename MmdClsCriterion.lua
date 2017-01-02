require 'MmdCriterion'
local MmdClsCriterion, Parent = torch.class('nn.MmdClsCriterion', 'nn.Criterion')

function MmdClsCriterion:__init()
  Parent.__init(self)
  self.mmdCriterion = nn.BCECriterion()
  self.clsCriterion = nn.CrossEntropyCriterion()
  self.gradInput = {self.mmdCriterion.gradInput, torch.Tensor()}
  self.lambda = 1
end

function MmdClsCriterion:updateOutput(input, target)
  local mmdLoss = self.mmdCriterion:updateOutput(input[1], target:ne(0):typeAs(target)) * self.lambda
  local clsLossMask = target:ne(0):nonzero():squeeze()
  local clsLoss = self.clsCriterion:updateOutput(input[2]:index(1, clsLossMask), target:index(1, clsLossMask))
  self.output = torch.Tensor({mmdLoss, clsLoss})
  return self.output
end

function MmdClsCriterion:updateGradInput(input, target)
  self.mmdCriterion:updateGradInput(input[1], target:ne(0):typeAs(target))
  self.mmdCriterion.gradInput:mul(self.lambda)
  if self.gradInput[1] ~= self.mmdCriterion.gradInput then
    self.gradInput[1] = self.mmdCriterion.gradInput
  end
  local clsLossMask = target:ne(0):nonzero():squeeze()
  self.gradInput[2]:resizeAs(input[2]):zero()
  self.gradInput[2]:indexCopy(1, clsLossMask, self.clsCriterion:updateGradInput(input[2]:index(1, clsLossMask), target:index(1, clsLossMask)))
  return self.gradInput
end
