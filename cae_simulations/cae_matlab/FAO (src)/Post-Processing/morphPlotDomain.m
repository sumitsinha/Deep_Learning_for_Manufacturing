% Plot domain of morphing mesh model 
function morphPlotDomain(data, idpart)

% data: input model
% idpart: part ID

% get part structure
[fp,flagp]=retrieveStructure(data.database, 'Part', idpart);

if ~flagp
    warning('Warning (morphing mesh) - part ID not valid!');
    return
end

% loop over all points
np=length(fp.Morphing);                     
for i=1:np
    
    Pc=fp.Morphing(i).Pc;
    
    %--
    % Control point
    plot3(Pc(1), Pc(2), Pc(3), 'o', 'parent',data.Axes3D.Axes,...
          'tag','tempobj','markerfacecolor','k','markersize',10)
      
   %--
   % Influence domain
   IDSelection=fp.Morphing(i).Selection;
   if IDSelection==0 % use automatic selection
       if fp.Status==0
           idnodes=data.database.Model.Nominal.Domain(idpart).Node;

           nodes=data.database.Model.Nominal.xMesh.Node.Coordinate(idnodes,:);
           dV=getBoundingVolume(nodes);
           dV.Type{1}=2;
           
%            plotSelection(data, dV)
       else
           warning('Warning (morphing mesh) - Part ID not active!');
       end
   else % use current selection
       [dV, flag]=retrieveStructure(data.database, 'Selection', IDSelection);
       
       if flag
            plotSelection(data, dV)
       else
            warning('Warning (morphing mesh) - Selection ID not valid!');
       end
   end       
end

