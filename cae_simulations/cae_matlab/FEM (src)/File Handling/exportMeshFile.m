function exportMeshFile(filename, fem, idpart)

% get file extension
[~,~,ext]=fileparts(filename);

% abaqus
if strcmp(ext,'.inp')
    disp('Writing Abaqus Mesh File...')
    exportInpFile(filename, fem, idpart)
    disp('Writing Abaqus Mesh File: completed!')
% nastran
elseif strcmp(ext,'.bdf') || strcmp(ext,'.dat')
    disp('Writing Nastran Mesh File...')
    exportBdfFile(filename, fem, idpart)
    disp('Writing Nastran Mesh File: completed!')
% stl format
elseif strcmp(ext,'.stl') || strcmp(ext,'.STL') 
    disp('Writing STL Mesh File...') 
    exportStlFile(filename, fem, idpart)
    disp('Writing STL Mesh File: completed!')
else
    disp('Mesh file not reconognised')
end

