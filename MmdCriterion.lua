require 'nn'
require 'utils.GaussianKernel'

local MmdCriterion, Parent = torch.class('nn.MmdCriterion', 'nn.Criterion')

function MmdCriterion:__init(gamma)
  self.gamma = gamma
  self.kernel = GaussianKernel(gamma)
end

function MmdCriterion:updateOutput(input, label)
  -- Source and target indices
  local sourceInput = input:index(1, label:ne(0):nonzero():squeeze())
  local targetInput = input:index(1, label:eq(0):nonzero():squeeze())

  -- Ensure that there are the same (even) number of source and target input
  local numSourceInput = sourceInput:size(1)
  local numTargetInput = targetInput:size(1)
  assert(numSourceInput == numTargetInput, 'source and target input must be same size')
  assert(numSourceInput % 2 == 0, 'there must be an even number of source and target input')

  -- Divide input into 4 tensors for quad-tuple based loss
  local sourceInputA = sourceInput:narrow(1, 1, numSourceInput / 2)
  local sourceInputB = sourceInput:narrow(1, numSourceInput / 2 + 1, numSourceInput / 2)
  local targetInputA = targetInput:narrow(1, 1, numTargetInput / 2)
  local targetInputB = targetInput:narrow(1, numTargetInput / 2 + 1, numTargetInput / 2)

  if not self.gamma or not self.kernel.gamma then
    local distances = torch.cat({sourceInputB - sourceInputA, targetInputA - sourceInputA, targetInputB - sourceInputA, targetInputA - sourceInputB, targetInputB - sourceInputB, targetInputB - targetInputA}, 1)
    local sqDistances = distances:pow(2)
    local gammas = torch.Tensor(distances:size(1))
    for i = 1, distances:size(1) do
      gammas[i] = torch.sum(sqDistances:select(1, i))
    end
    self.gamma = gammas:median():squeeze()
    self.kernel.gamma = self.gamma
  end

  -- Calculate kernel function using quad-tuples
  local k1 = self.kernel:forward(sourceInputA, sourceInputB)
  local k2 = self.kernel:forward(targetInputA, targetInputB)
  local k3 = self.kernel:forward(sourceInputA, targetInputA)
  local k4 = self.kernel:forward(sourceInputB, targetInputB)
  local k = k1 + k2 - k3 - k4

  self.output = torch.mean(k)
  return self.output
end

function MmdCriterion:updateGradInput(input, label)
  -- Source and target indices
  local sourceInput = input:index(1, label:ne(0):nonzero():squeeze())
  local targetInput = input:index(1, label:eq(0):nonzero():squeeze())

  -- Divide input into 4 tensors for quad-tuple based loss
  local numSourceInput = sourceInput:size(1)
  local numTargetInput = targetInput:size(1)
  local sourceInputA = sourceInput:narrow(1, 1, numSourceInput / 2)
  local sourceInputB = sourceInput:narrow(1, numSourceInput / 2 + 1, numSourceInput / 2)
  local targetInputA = targetInput:narrow(1, 1, numTargetInput / 2)
  local targetInputB = targetInput:narrow(1, numTargetInput / 2 + 1, numTargetInput / 2)

  -- Calculate gradients of kernels
  local k1GradSourceA, k1GradSourceB = self.kernel:backward(sourceInputA, sourceInputB)
  local k2GradTargetA, k2GradTargetB = self.kernel:backward(targetInputA, targetInputB)
  local k3GradSourceA, k3GradTargetA = self.kernel:backward(sourceInputA, targetInputA)
  local k4GradSourceB, k4GradTargetB = self.kernel:backward(sourceInputB, targetInputB)

  -- Compose kernel gradients into source/target gradients
  local gradSourceA = (k1GradSourceA - k3GradSourceA) / (numSourceInput)
  local gradSourceB = (k1GradSourceB - k4GradSourceB) / (numSourceInput)
  local gradTargetA = (k2GradTargetA - k3GradTargetA) / (numTargetInput)
  local gradTargetB = (k2GradTargetB - k4GradTargetB) / (numTargetInput)

  self.gradInput = torch.cat({gradSourceA, gradSourceB, gradTargetA, gradTargetB}, 1)
  return self.gradInput
end
