% save large large data DCT structure

function saveLargeFileDCT(data, filepath)

disp('saving model...')
nd=length(data);

datasave=[];
for i=1:nd
    
    datasave(i).NoEmpVoxelId=data(i).NoEmpVoxelId;
    datasave(i).VoxelId=data(i).VoxelId;
    datasave(i).CoeffInv=data(i).CoeffInv;
    datasave(i).BBox=data(i).BBox;
    
end

% save
for i=1:nd
    
    filemat=sprintf('%s%spart_ID_%g.mat', filepath, '\', i);
    
    datai=datasave(i);
    fprintf('    saving: %s\n', filemat)  
    save(filemat, 'datai')
    
end

