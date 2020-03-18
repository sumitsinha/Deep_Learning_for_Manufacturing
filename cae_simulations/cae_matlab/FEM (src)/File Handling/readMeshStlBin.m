% This function reads an STL file in binary format
function [face, vertex] = readMeshStlBin(filename)

% INPUT
% filename: filename

% OUTPUT
% face: element connection (tria elements)
% vertex: xyz coordinates

%--
fid=fopen(filename, 'r'); %Open the file, assumes STL Binary format.
if fid == -1 
    return
end

ftitle=fread(fid,80,'uchar=>schar'); % Read file title
numFaces=fread(fid,1,'int32'); % Read number of Faces

T = fread(fid,inf,'uint8=>uint8'); % read the remaining values
fclose(fid);

% Each facet is 50 bytes
%  - Three single precision values specifying the face normal vector
%  - Three single precision values specifying the first vertex (XYZ)
%  - Three single precision values specifying the second vertex (XYZ)
%  - Three single precision values specifying the third vertex (XYZ)
%  - Two color bytes (possibly zeroed)

% 3 dimensions x 4 bytes x 4 vertices = 48 bytes for triangle vertices
% 2 bytes = color (if color is specified)

trilist = 1:48;

ind = reshape(repmat(50*(0:(numFaces-1)),[48,1]),[1,48*numFaces])+repmat(trilist,[1,numFaces]);
Tri = reshape(typecast(T(ind),'single'),[3,4,numFaces]);

vertex=Tri(:,2:4,:);
vertex = reshape(vertex,[3,3*numFaces]);
vertex = double(vertex)';

face = reshape(1:3*numFaces,[3,numFaces])';

% remove duplicates
[vertex, ~, indexn] =  unique(vertex, 'rows');
face = indexn(face);

fprintf('    Mesh Summary - no. of nodes: %g\n', size(vertex,1));
fprintf('    Mesh Summary - no. of TRIA: %g\n',size(face,1));

