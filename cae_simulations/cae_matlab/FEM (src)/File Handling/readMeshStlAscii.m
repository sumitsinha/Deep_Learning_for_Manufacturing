% This function reads an STL file in ascii format
function [face, vertex]=readMeshStlAscii(filename)
 
% % INPUT
% % filename: filename
% 
% % OUTPUT
% % face: element connection (tria elements)
% % vertex: xyz coordinates
% 
% % NB: duplicated vertices are automatically deleted
% %----------------------------------

% set intial values:
vertex=[];
face=-1;
    
% read file
fid=fopen(filename,'r');

% check
if fid==-1
    return
end

% scan file
fmt = '%*s %*s %f32 %f32 %f32 \r\n %*s %*s \r\n %*s %f32 %f32 %f32 \r\n %*s %f32 %f32 %f32 \r\n %*s %f32 %f32 %f32 \r\n %*s \r\n %*s \r\n';
C=textscan(fid, fmt, 'HeaderLines', 1);
fclose(fid);

% extract vertices
v1 = cell2mat(C(4:6));
v2 = cell2mat(C(7:9));
v3 = cell2mat(C(10:12));

if isnan(C{4}(end))
    v1 = v1(1:end-1,:); % strip off junk from last line
    v2 = v2(1:end-1,:); % strip off junk from last line
    v3 = v3(1:end-1,:); % strip off junk from last line
end

vtemp = [v1 v2 v3]';

if isempty(vtemp)      
    fprintf('    Mesh Summary - no. of nodes: %g\n', 0);
    fprintf('    Mesh Summary - no. of TRIA: %g\n',0);
    return
end

vertex = zeros(3,numel(vtemp)/3);
vertex(:) = vtemp(:);
vertex = vertex';

%  reshape output face
listVertex=1:size(vertex,1);
face=reshape(listVertex',3,length(listVertex)/3)';

% remove duplicates
[vertex, ~, indexn] =  unique(vertex, 'rows');
face = indexn(face);   

fprintf('    Mesh Summary - no. of nodes: %g\n', size(vertex,1));
fprintf('    Mesh Summary - no. of TRIA: %g\n',size(face,1));




