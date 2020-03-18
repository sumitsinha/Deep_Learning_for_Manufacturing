%--
function p=modelLoadInputFile(filepath, maxcol, checkmaxcol)

if nargin==2
    checkmaxcol=false;
end

p=[];    
% check-out the file
try
    d=importdata(filepath);
    if iscell(d) || isstruct(d)
        error('Wrong file format @%s', filepath)
    else
        
        % run control on number of columns
        if checkmaxcol
            nc=size(d,2);
            if nc~=maxcol
                error('Wrong file format @%s', filepath)
            end
        end
    end
    
    p=d;
    
catch %#ok<CTCH>
    error('Wrong file format @%s', filepath)
end
