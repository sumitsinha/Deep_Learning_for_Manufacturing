% load large large data structure

function fem=loadLargeFile(fem, filepath)

disp('load model...')
fie=fieldnames(fem);

for i=1:length(fie)
    
    filemat=[filepath, '\', fie{i},'.mat'];
    
    fprintf('    loading: %s\n', fie{i})  
    var=load(filemat);
    
    f=fieldnames(var);
    
    fem=setfield(fem, fie{i}, getfield(var,f{1}));
    
end