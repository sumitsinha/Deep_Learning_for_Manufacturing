% plot normals of input data structure
function plotDataInputSingleNormal(data, field, id, paraid, tag)

% data: data structure
% field: field to plot
% id: ID
% paraid: parameter id
% tag: graphic tag
        
% get field
[f, flag]=retrieveStructure(data.database, field, id);

if ~flag
    return
end

%--
if ~f.Graphic.ShowNormal
    return
end
%--

% check status
[flag, idpoint]=checkInputStatus(f, paraid, field);

% define tag for plotting purposes
if nargin<5
    tag=getTagValue(field, id);
end

for i=1:length(flag)
     plotNormalItem(data, f, flag(i), field, idpoint(i), paraid(i), tag)
end
%------------------------------

% rendering functions
function plotNormalItem(data, f, flag, root, idpoint, id, tag)
      
if flag && f.Enable % green
           
    if strcmp(root,'Stitch') || strcmp(root,'Dimple')
        % start
        if f.Parametrisation.Geometry.Type{1}{1}==1 % ref
            ids=1;
        else
           if id>size(f.Pam{1},1)
                ids=size(f.Pam{1},1);
            else
                ids=id;
            end
        end

        if strcmp(root,'Stitch') 
            % end
            if f.Parametrisation.Geometry.Type{2}{1}==1 % ref
                ide=1;
            else
                if id>size(f.Pam{2},1)
                    ide=size(f.Pam{2},1);
                else
                    ide=id;
                end
            end
        
            if f.Type{1}==1  % linear  
                
                Pm=[f.Pam{1}(ids,:)
                    f.Pam{2}(ide,:)]; 
                Nm=[f.Nam{1}(ids,:)
                    f.Nam{2}(ide,:)]; 
            elseif f.Type{1}==2 || f.Type{1}==3 % circular/rigid link
                Pm=f.Pam{1}(ids,:); 
                Nm=f.Nam{1}(ids,:); 
            elseif f.Type{1}==4 % edge
                Pm=[f.Pam{1}(ids,:)
                    f.Pam{2}(ide,:)
                    f.Pm(3,:)]; 
                [idKnots, flagknots]=boundaryBy3Points(data.database.Model.Nominal, Pm, f.SearchDist(1));

                if ~flagknots
                    return
                end
    
                Pm=data.database.Model.Nominal.xMesh.Node.Coordinate(idKnots,:);
                Nm=data.database.Model.Nominal.xMesh.Node.Normal(idKnots,:);
        
            end
        elseif strcmp(root,'Dimple') 
                Pm=f.Pam{1}(ids,:); 
                Nm=f.Nam{1}(ids,:); 
        end
        
    else % others

        if f.Parametrisation.Geometry.Type{1}{1}==1 % ref
            id=1;
        end
        
        Pm=f.Pam{idpoint}(id,:);
        Nm=f.Nam{idpoint}(id,:);
    end
      
    %--
    lsymbol=data.Axes3D.Options.LengthAxis;
    ax=data.Axes3D.Axes;
    renderAxis(Nm, Pm, ax, lsymbol, tag,'k');
 
end
