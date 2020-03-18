% save large large data structure

function saveLargeFile(fem, filepath)

disp('saving model...')
fie=fieldnames(fem);

for i=1:length(fie)
    
    filemat=[filepath, '\', fie{i},'.mat'];
    var=getfield(fem, fie{i});
    
    fprintf('    saving: %s\n', fie{i})  
    save(filemat, 'var')
    
end