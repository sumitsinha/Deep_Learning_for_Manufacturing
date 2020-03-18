% Split QUAD element into TRIA element
function tria=splitQuad2Tria(quad, dirSplit)

% direction of split (1: secondary diagonal; 2: main diagonal)
    % 1 => [1 2 3 4] => [1 2 3; 3 4 1]
    % 2 => [1 2 3 4] => [1 4 2; 2 4 3]

% check input
if nargin==1
    dirSplit=1;
end

% split element
n=size(quad,1);
tria=zeros(2*n,3);
c=1;
for i=1:n
    quadi=quad(i,:);
    
    if dirSplit==1 % secondary diagonal
        tria(c,:)=quadi([1 2 3]);
        tria(c+1,:)=quadi([3 4 1]);
    elseif dirSplit==2 % main diagonal
        tria(c,:)=quadi([1 4 2]);
        tria(c+1,:)=quadi([2 4 3]);
    end
    c=c+2;
end
