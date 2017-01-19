local M = {
  Runner = require 'runners.runner',
}
local Tester = torch.class('Tester', 'Runner', M)

function Tester:__init(model, criterion, vggFeatures, opt, logger, setName)
  self.model = model
  self.criterion = criterion
  self.vggFeatures = vggFeatures
  self.opt = opt
  self.logger = logger
  self.setName = setName or 'Test'
end

function Tester:test(epoch, dataloader)
  local outdir = paths.concat(self.opt.save, 'imgs', '' .. (epoch or 'final'))
  if not paths.dirp(outdir) then
    local res = os.execute('mkdir -p ' .. outdir)
    assert(res, 'Could not open or make save directory ' .. outdir)
  end

  local timer = torch.Timer()
  local dataTimer = torch.Timer()
  local size = dataloader:size()

  local nCrops = self.opt.tenCrop and 10 or 1
  local lossSum, timeSum = 0.0, 0.0
  local N = 0

  self.model:evaluate()
  for n, sample in dataloader:run() do
    local dataTime = dataTimer:time().real

    -- Copy input and target to the GPU
    local features = {}
    self:copyInputs(sample)
    self.vggFeatures:forward(self.input)
    for i, v in ipairs(self.vggFeatures.output) do
      table.insert(features, v:clone())
    end
    local output = self.model:forward(features):float()
    local batchSize = output:size(1) / nCrops
    local loss = self.criterion:forward(output, features)

    local time = timer:time().real
    lossSum = lossSum + loss*batchSize
    timeSum = timeSum + time
    N = N + batchSize

    local epochString = epoch and ('[' .. epoch .. ']') or ''
    print((' | %s: %s[%d/%d]   Time %.3f  Data %.3f  Err %.3f'):format(
      self.setName, epochString, n, size, time, dataTime, loss))

    for i = 1, #sample.path do
      image.save(paths.concat(outdir, sample.path[i]), output[i])
    end

    timer:reset()
    dataTimer:reset()
  end
  self.model:training()

  return {
    time = timeSum,
    loss = lossSum / N,
  }
end

return M.Tester
