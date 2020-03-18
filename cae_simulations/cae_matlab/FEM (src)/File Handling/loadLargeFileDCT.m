% load large large data DCT structure

function data=loadLargeFileDCT(filepath, nd)

disp('loading model...')

for i=1:nd
    
    filemat=sprintf('%s%spart_ID_%g.mat', filepath, '\', i);
    
    fprintf('    loading: %s\n', filemat)  
    
    var=load(filemat);

    data(i)=var.datai;
    
end