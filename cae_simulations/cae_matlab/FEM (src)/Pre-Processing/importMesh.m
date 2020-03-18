% import mesh file
function fem=importMesh(fem, filename)

% get file extension
[~,~,ext]=fileparts(filename);

% abaqus
if strcmp(ext,'.inp')
    disp('Reading Abaqus Mesh File...')
    [quad,tria,node]=readMeshAbaqus(filename); % # MEX function
    
% nastran
elseif strcmp(ext,'.bdf') || strcmp(ext,'.dat')
    disp('Reading Nastran Mesh File...')
    [quad,tria,node]=readMeshNas(filename); % # MEX function
    
% stl format
elseif strcmp(ext,'.stl') || strcmp(ext,'.STL') 
    disp('Reading STL Mesh File...')
    
    stltype = getStlFormat(filename);
    
    if strcmp(stltype,'ascii')
        
        quad=-1; % only trias elements are supported
        [tria,node]=readMeshStlAscii(filename); % matlab function
        
    elseif strcmp(stltype,'binary')
        
        quad=-1; % only trias elements are supported
        [tria, node]=readMeshStlBin(filename); % matlab function
        
    end
    
else
    disp('Mesh file not reconognised')
    return
end

% save all
fem=femSaveMesh(fem, quad, tria, node);





