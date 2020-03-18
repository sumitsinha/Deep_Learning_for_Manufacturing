function varargout=countUnique(V, mode)

% V: double or cell array
% mode: "row", "col" => scan by row / scan by col

% b: index of duplications
% bu: unique index of duplications
% nvar: no. of unique items

% Example:
% V=[1 2 1 3 2 7 1];
% [b, nvar]=myunique(a, 'col');
% b=[1 2 1 3 2 4 1];
% nvar=4;

if nargin==1
    mode='row';
end

% init output
if strcmp(mode, 'row')
    n=size(V,1);
    b=zeros(n, 1);
elseif strcmp(mode, 'col')
    n=size(V,2);
    b=zeros(1,n);
end

nvar=0;
for i=1:n-1
    
    if b(i)==0
        nvar=max(b)+1;
        b(i)=nvar;
    end
    
    for j=i+1:n
                
        if strcmp(mode, 'row') 
            vi=V(i,:);
            vj=V(j,:);
        elseif strcmp(mode, 'col') 
            vi=V(:,i);
            vj=V(:,j);
        end
        
        if localcheckequal(vi,vj)
            b(j)=b(i);
        end
    end
end

if b(end)==0
    nvar=max(b)+1;
    b(end)=nvar;
end

bu=unique(b);

if nargout==1
    varargout{1}=b;
end

if nargout==2
    varargout{1}=b;
    varargout{2}=bu;
end

if nargout==3
    varargout{1}=b;
    varargout{2}=bu;
    varargout{3}=nvar;
end


%--
function flag=localcheckequal(a,b)

flag=true;
n=length(a);
for i=1:n
    
    if iscell(a) % cell array
        if ischar(a{i}) && ischar(b{i})
            if ~strcmp(a{i}, b{i}) % string
                flag=false;
                return
            end
        else
            if a{i}~=b{i} % double
                flag=false;
                return
            end            
        end

    else % double array
        if a(i)~=b(i)
            flag=false;
            return
        end
    end
end

 
 


