local Filenames = {}



local experimentDir = os.getenv('EXPERIMENT_DIR')
assert(experimentDir, 'EXPERIMENT_DIR variable unset')
if not paths.dirp(experimentDir) and not paths.mkdir(experimentDir) then
  assert('error: unable to create checkpoint directory: ' .. experimentDir .. '\n')
end

function Filenames.model()
  return paths.concat(experimentDir, 'model.t7')
end

function Filenames.optimState()
  return paths.concat(experimentDir, 'optimState.t7')
end

function Filenames.latest()
  return paths.concat(experimentDir, 'latest.t7')
end

return Filenames
