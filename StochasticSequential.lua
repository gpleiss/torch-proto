local StochasticSequential, Parent = torch.class('nn.StochasticSequential', 'nn.Sequential')

function StochasticSequential:__init(p)
  Parent.__init(self)
  self.p = p
end

function StochasticSequential:updateOutput(input)
  Parent.updateOutput(self, input)
  local randVal = torch.uniform()
  if self.train and randVal < self.p then
    self.output:fill(0)
    self.skipped = true
  end
  return self.output
end

function StochasticSequential:backward(input, gradOutput, scale)
  if self.train and self.skipped then
    self.gradInput:resizeAs(input):fill(0)
  else
    Parent.backward(self, input, gradOutput, scale)
  end
  self.skipped = false
  return self.gradInput
end

function StochasticSequential:updateGradInput(input, gradOutput)
  if self.train and self.skipped then
    self.gradInput:resizeAs(input):fill(0)
  else
    Parent.updateGradInput(self, input, gradOutput)
  end
  self.skipped = false
  return self.gradInput
end

function StochasticSequential:accGradParameters(input, gradOutput, scale)
  if self.train and self.skipped then
    self.gradInput:resizeAs(input):fill(0)
  else
    Parent.accGradParameters(self, input, gradOutput, scale)
  end
  self.skipped = false
  return self.gradInput
end

function StochasticSequential:accUpdateGradParameters(input, gradOutput, lr)
  if self.train and self.skipped then
    self.gradInput:resizeAs(input):fill(0)
  else
    Parent.accUpdateGradParameters(self, input, gradOutput, lr)
  end
  self.skipped = false
  return self.gradInput
end
