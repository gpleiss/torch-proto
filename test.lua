require 'runner'
local Tester = torch.class('Tester', 'Runner')

function Tester:__init(model, opt, logger, setName)
  self.model = model
  self.opt = opt
  self.logger = logger
  self.setName = setName or 'Test'
end

function Tester:test(epoch, dataloader)
  -- Computes the top-1 and top-5 err on the validation set

  local timer = torch.Timer()
  local dataTimer = torch.Timer()
  local size = dataloader:size()

  local nCrops = self.opt.tenCrop and 10 or 1
  local top1Sum, top5Sum, timeSum = 0.0, 0.0, 0.0
  local N = 0

  self.model:evaluate()
  for n, sample in dataloader:run() do
    local dataTime = dataTimer:time().real

    -- Copy input and target to the GPU
    self:copyInputs(sample)

    local output = self.model:forward(self.input):float()
    local batchSize = output:size(1) / nCrops

    local top1, top5 = self:computeScore(output, sample.target, nCrops)
    local time = timer:time().real
    top1Sum = top1Sum + top1*batchSize
    top5Sum = top5Sum + top5*batchSize
    timeSum = timeSum + time
    N = N + batchSize

    local epochString = epoch and ('[' .. epoch .. ']') or ''
    print((' | %s: %s[%d/%d]   Time %.3f  Data %.3f  top1 %7.3f (%7.3f)  top5 %7.3f (%7.3f)'):format(
      self.setName, epochString, n, size, time, dataTime, top1, top1Sum / N, top5, top5Sum / N))

    timer:reset()
    dataTimer:reset()
  end
  self.model:training()

  if epoch then
    print((' * Finished epoch # %d    top1: %7.3f  top5: %7.3f\n'):format(
      epoch, top1Sum / N, top5Sum / N))
  else
    print((' Results: top1: %7.3f  top5: %7.3f\n'):format(top1Sum / N, top5Sum / N))
  end

  return {
    top1 = top1Sum / N,
    top5 = top5Sum / N,
    time = timeSum,
  }
end
