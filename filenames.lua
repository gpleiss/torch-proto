local Filenames = {}


local dataDir = os.getenv('DATA_DIR')
assert(dataDir, 'DATA_DIR variable unset')
if not paths.dirp(dataDir) then
  assert('error: unable to find data directory: ' .. dataDir .. '\n')
end

local experimentDir = os.getenv('EXPERIMENT_DIR')
assert(experimentDir, 'EXPERIMENT_DIR variable unset')
if not paths.dirp(experimentDir) and not paths.mkdir(experimentDir) then
  assert('error: unable to create checkpoint directory: ' .. experimentDir .. '\n')
end

Filenames.dataDir = dataDir
Filenames.experimentDir = experimentDir



function Filenames.model()
  return paths.concat(experimentDir, 'model.t7')
end

function Filenames.optimState()
  return paths.concat(experimentDir, 'optimState.t7')
end

function Filenames.logger()
  return paths.concat(experimentDir, 'logger.txt')
end

function Filenames.latest()
  return paths.concat(experimentDir, 'latest.t7')
end

function Filenames.cacheFile(dataset)
  return paths.concat(Filenames.data(dataset), dataset .. '.t7')
end

function Filenames.data(dataset, split)
  if split then
    return paths.concat(dataDir, dataset, split)
  else
    return paths.concat(dataDir, dataset)
  end
end

return Filenames
