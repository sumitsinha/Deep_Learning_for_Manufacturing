% calculate element to element connectivity
function fem=femElement2Element(fem)

% no. of element and nodes
nele=length(fem.xMesh.Element);

% set initial values
fem.Sol.Element2Element=cell(1,nele);

% loop over elements
for i=1:nele
    
    facek=fem.xMesh.Element(i).Element;
    
    % calculate the counter (in order to speed-up the computation)
    count=0;
    for j=1:length(facek)
        count=count + length(fem.Sol.Node2Element{facek(j)});
    end
    
    % then... allocate values
    temp=zeros(1,count);
    ce=0;
    for j=1:length(facek)
        cs=ce+1;       
        ce=ce+length(fem.Sol.Node2Element{facek(j)});
                
        temp(cs:ce)=fem.Sol.Node2Element{facek(j)};
    end
    
    % remove duplicates
    fem.Sol.Element2Element{i}=unique(temp);
    
end

