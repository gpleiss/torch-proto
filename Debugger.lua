require 'nn'

local Debugger = torch.class('nn.Debugger', 'nn.Module')

function Debugger:__init(str)
  self.str = str
end

function Debugger:updateOutput(input)
  self.output = input
  print('At ' .. self.str .. ':')
  if type(self.output) == 'table' then
    print(self.output)
  else
    print({self.output})
  end
  return self.output
end

function Debugger:updateGradInput(input, gradOutput)
  self.gradInput = gradOutput
  print('At ' .. self.str .. ':')
  if type(self.gradInput) == 'table' then
    print(self.gradInput)
  else
    print({self.gradInput})
  end
  return self.gradInput
end
