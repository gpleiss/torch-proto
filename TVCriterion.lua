local TVCriterion, Parent = torch.class('nn.TVCriterion', 'nn.Criterion')

function TVCriterion:__init(beta, eps)
  self.beta = beta or 2
  self.eps = eps or 1e-3

  self.xDiff = torch.Tensor()
  self.yDiff = torch.Tensor()
  self.sqDiffNorm = torch.Tensor()

  Parent.__init(self)
end

function TVCriterion:updateOutput(input)
  self.xDiff:resizeAs(input:sub(1, -1, 1, -1, 1, -2, 1, -2))
  self.yDiff:resizeAs(input:sub(1, -1, 1, -1, 1, -2, 1, -2))
  self.sqDiffNorm:resizeAs(input:sub(1, -1, 1, -1, 1, -2, 1, -2)):zero()

  -- Compute diffs
  torch.add(self.xDiff, input:sub(1, -1, 1, -1, 1, -2, 1, -2), -1, input:sub(1, -1, 1, -1, 1, -2, 2, -1))
  torch.add(self.yDiff, input:sub(1, -1, 1, -1, 1, -2, 1, -2), -1, input:sub(1, -1, 1, -1, 2, -1, 1, -2))

  -- Compute sqrare of diffs norm
  self.sqDiffNorm:addcmul(self.xDiff, self.xDiff)
  self.sqDiffNorm:addcmul(self.yDiff, self.yDiff)

  -- Clamp values by eps
  self.sqDiffNorm:clamp(self.eps, math.huge)

  -- Raise square diff norm to the beta/2 power, and sum
  self.output = torch.norm(self.sqDiffNorm, self.beta / 2.0) ^ (self.beta / 2.0)

  return self.output
end

function TVCriterion:updateGradInput(input, target)
  self.gradInput:resizeAs(input):zero()

  -- Compute derivative of square diff norm
  self.sqDiffNorm:pow(self.beta / 2.0 - 1):mul(self.beta / 2.0)

  -- Compute diff gradients
  self.xDiff:cmul(self.sqDiffNorm):mul(2)
  self.yDiff:cmul(self.sqDiffNorm):mul(2)

  -- Fill out self.gradInput
  torch.add(self.gradInput:sub(1, -1, 1, -1, 1, -2, 1, -2), self.xDiff, self.yDiff)
  self.gradInput:sub(1, -1, 1, -1, 1, -2, 2, -1):csub(self.xDiff)
  self.gradInput:sub(1, -1, 1, -1, 2, -1, 1, -2):csub(self.yDiff)

  return self.gradInput
end
