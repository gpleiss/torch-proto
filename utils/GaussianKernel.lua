local GaussianKernel = torch.class('GaussianKernel')

function GaussianKernel:__init(gamma)
  self.gamma = gamma
end

function GaussianKernel:forward(x1, x2)
  local sqDiff = torch.pow(x1 - x2, 2)
  while sqDiff:nDimension() > 1 do
    sqDiff = torch.sum(sqDiff, 2)
    sqDiff = sqDiff:squeeze(2)
  end

  return torch.exp(-sqDiff / self.gamma)
end

function GaussianKernel:backward(x1, x2)
  local forwardResVals = self:forward(x1, x2)
  local forwardRes = torch.Tensor(x1:size()):typeAs(x1)
  for i = 1, forwardResVals:size(1) do
    forwardRes[i]:fill(forwardResVals[i])
  end

  local grad = torch.cmul((x1 - x2), forwardRes) * -2 / self.gamma
  return grad, -grad
end
