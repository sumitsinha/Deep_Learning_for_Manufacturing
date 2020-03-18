% Compute part-to-part gaps between parts in free state condition
function GAP=modelGetGapParts(data)

% Inputs:
% data: model

% Outputs:    
% GAP: gap field 

% Set simulation model
fem=data.Model.Nominal;
fem.Options.GapFrame='ref'; % Use the geometric distance/reference using geometry stored in "fem.xMesh"
%
% Reset any contact pair
fem.Boundary.ContactPair=[];

% Assign contact pair to the model
field='Contact';
f=getInputFieldModel(data, field);
n=length(f);  
count=1;
for i=1:n
    fi=f(i);
    if fi.Status{1}==0   
        fem.Boundary.ContactPair(count).Master=fi.Master;
        if fi.MasterFlip
             fem.Boundary.ContactPair(count).MasterFlip=true;
        else
             fem.Boundary.ContactPair(count).MasterFlip=false;
        end
        fem.Boundary.ContactPair(count).Slave=fi.Slave;
        fem.Boundary.ContactPair(count).SearchDist=fi.SearchDist(1);
        fem.Boundary.ContactPair(count).SharpAngle=fi.SearchDist(2); 
        fem.Boundary.ContactPair(count).Offset=fi.Offset;  
        if fi.Use  
             fem.Boundary.ContactPair(count).Enable=true;
        else % use it only for post-processing purposes but not for constraint solving
            fem.Boundary.ContactPair(count).Enable=false;
        end
        fem.Boundary.ContactPair(count).Sampling=fi.Sampling;
        fem.Boundary.ContactPair(count).Frame='ref';
        count=count+1;
    end
end
%
% Compute gaps
nctpairs=length(fem.Boundary.ContactPair);
nnode=size(fem.xMesh.Node.Coordinate,1);
if nctpairs>0
    for i=1:nctpairs
       fem.Sol.Gap(i).Gap=zeros(1,nnode); % gap for each contact pair
       
       fem.Sol.Gap(i).max=fem.Options.Min;
       fem.Sol.Gap(i).min=fem.Options.Max;  
    end
end
fem=getGapsVariable(fem);

% Save back
GAP=fem.Sol.Gap;
