%-----------------
function [tria, quad, node, flag]=readMesh(filename)

flag=true;
tria=[];
quad=[];
node=[];

if isempty(filename)
    flag=false;
    return
end

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
        
        [tria,node]=readMeshStlAscii(filename); % matlab function
        
    elseif strcmp(stltype,'binary')
        
        [tria, node]=readMeshStlBin(filename); % matlab function
        
    end
    
else
    flag=false;
    disp('Mesh file not reconognised')
    return
end
