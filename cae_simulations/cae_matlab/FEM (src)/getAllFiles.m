function fileList = getAllFiles(dirName, mode)

% dirName: folder name
% mode: 'full', 'file'

filet=extract_files(dirName);

if strcmp(mode,'full')
    fileList=filet;
elseif strcmp(mode,'file')
    n=length(filet);
    
    fileList=cell(n,1);
    for i=1:n
        [~,name, ext] = fileparts(filet{i});
        fileList{i}=[name,ext];
    end
end

%---
function fileList=extract_files(dirName)

  dirData = dir(dirName);      %# Get the data for the current directory
  dirIndex = [dirData.isdir];  %# Find the index for directories
  fileList = {dirData(~dirIndex).name}';  %'# Get a list of the files
  if ~isempty(fileList)
    fileList = cellfun(@(x) fullfile(dirName,x),...  %# Prepend path to files
                       fileList,'UniformOutput',false);
  end
  subDirs = {dirData(dirIndex).name};  %# Get a list of the subdirectories
  validIndex = ~ismember(subDirs,{'.','..'});  %# Find index of subdirectories
                                               %#   that are not '.' or '..'
  for iDir = find(validIndex)                  %# Loop over valid subdirectories
    nextDir = fullfile(dirName,subDirs{iDir});    %# Get the subdirectory path
    fileList = [fileList; extract_files(nextDir)];  %# Recursively call getAllFiles
  end
  
  
