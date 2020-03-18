% Get list of parameters in the model 
function data=modelGetParameters(data)

% Input:
% data: input model

% Output:
% data: udpated data model with updated "data.Assembly.Parameter", with the following entries
    % "Placement" parameter => rigid part placement parameters
            % [mode, partID, parameter type, reference]
                % {1} mode: 1="Placement" parameter
                % {2} groupID: group ID
                % {3} partID: part ID
                % {4} parameter type: 1/2/3/4/5/6 => alfa, beta, gamma, T, V, N
                % {5} parameter name: [alfa, beta, gamma, T, V, N]
                % {6} reference: 0/1 => global UCS, local UCS to partID
                % {7} NOT IN USE
    % "Position" parameter => parameters related to the position of input items(locators, etc.)
        % [mode, groupID, field, fieldID, pointID, parameter type, parameter name]   
            % {1} mode: 2="Position" parameter
            % {2} group ID
            % {3} Field name ('Stitch', 'Hole', 'Slot', 'NcBlock', 'ClampS', 'ClampM', 'CustomConstraint')
            % {4} Field ID
            % {5} Point ID (in case of "stitch", the geometry is defined by 2 points. So "point ID" may be either 1 or 2)
            % {6} parameter type: 2/3/4/5/6/7/8/9/10 (T, V, N, TV, TN, VN, TVN, u, v)
            % {7} parameter name (Reference, T, V,... TVN, u, v)
%----------------------

% STEP 1: get "Placement" parameters
data=modelGetParametersPlacement(data);

% STEP 2: get "Position" parameters
data=modelGetParametersPosition(data);

%----------------------
% Get list of parameters in the model @Placement
function data=modelGetParametersPlacement(data)

% get fields
p=getInputFieldModel(data, 'Part');

% loop over all parts
c=1;
dv=cell(0,7);
for i=1:length(p)
    pi=p(i);
    for j=1:length(pi.Parametrisation.Type) % ['alfa', 'beta', 'gamma', 'T', 'V', 'N']
        if pi.Parametrisation.Type(j)~=0
            dv{c, 1}=1;
            dv{c, 2}=1;
            dv{c, 3}=i;
            dv{c, 4}=j;
            dv{c, 5}=pi.Parametrisation.Name{j};
            dv{c, 6}=pi.Parametrisation.UCS;
            dv{c, 7}=0;
            c=c+1;
        end
    end
end
%
% udpate and save back
data.Assembly.Parameter=[data.Assembly.Parameter
                         dv];                    
%
% Get list of parameters in the model @Position
function data=modelGetParametersPosition(data)
    
% define fields
fields={'Stitch', 'Hole', 'Slot', 'NcBlock', 'ClampS', 'ClampM', 'CustomConstraint'};

% run over all fields
c=1;
dv=cell(0,7);
for k=1:length(fields)
    
    field=fields{k};

    f=getInputFieldModel(data, field);
    n=length(f);
    for id=1:n
      [fid, ~]=retrieveStructure(data, field, id); 
      
      fgeom=fid.Parametrisation.Geometry;
      
      np=length(fgeom.Type); % no. of points
      
      if strcmp(field,'Stitch')
          if fid.Type{1}==2 || fid.Type{1}==3 % circular; rigid link
             np=1;
          end
      end
      
      for j=1:np
          
          [flagactive, ~]=checkInputStatusGivenPoint(fid, 1, j);
          
          % check if it is active          
          if fid.Enable && flagactive
              
            ty=fgeom.Type{j}{1};
            
            if ty>1 % parametric
                if ty==2 || ty==3 || ty==4 || ty==9 || ty==10
                    dv{c, 1}=2;
                    dv{c, 2}=1;
                    dv{c, 3}=field;
                    dv{c, 4}=id;
                    dv{c, 5}=j;
                    dv{c, 6}=ty;
                    dv{c, 7}=fgeom.Type{j}{ty+1};
                    c=c+1;
                elseif ty==5
                    dv{c, 1}=2;
                    dv{c, 2}=1;
                    dv{c, 3}=field;
                    dv{c, 4}=id;
                    dv{c, 5}=j;
                    dv{c, 6}=ty;
                    dv{c, 7}='T';
                    c=c+1;
                    
                    dv{c, 1}=2;
                    dv{c, 2}=1;
                    dv{c, 3}=field;
                    dv{c, 4}=id;
                    dv{c, 5}=j;
                    dv{c, 6}=ty;
                    dv{c, 7}='V';
                    c=c+1;
                elseif ty==6
                    dv{c, 1}=2;
                    dv{c, 2}=1;
                    dv{c, 3}=field;
                    dv{c, 4}=id;
                    dv{c, 5}=j;
                    dv{c, 6}=ty;
                    dv{c, 7}='T';
                    c=c+1;
                    
                    dv{c, 1}=2;
                    dv{c, 2}=1;
                    dv{c, 3}=field;
                    dv{c, 4}=id;
                    dv{c, 5}=j;
                    dv{c, 6}=ty;
                    dv{c, 7}='N';
                    c=c+1;
                 elseif ty==7
                    dv{c, 1}=2;
                    dv{c, 2}=1;
                    dv{c, 3}=field;
                    dv{c, 4}=id;
                    dv{c, 5}=j;
                    dv{c, 6}=ty;
                    dv{c, 7}='V';
                    c=c+1;
                    
                    dv{c, 1}=2;
                    dv{c, 2}=1;
                    dv{c, 3}=field;
                    dv{c, 4}=id;
                    dv{c, 5}=j;
                    dv{c, 6}=ty;
                    dv{c, 7}='N';
                    c=c+1;
                 elseif ty==8
                    dv{c, 1}=2;
                    dv{c, 2}=1;
                    dv{c, 3}=field;
                    dv{c, 4}=id;
                    dv{c, 5}=j;
                    dv{c, 6}=ty;
                    dv{c, 7}='T';
                    c=c+1;
                    
                    dv{c, 1}=2;
                    dv{c, 2}=1;
                    dv{c, 3}=field;
                    dv{c, 4}=id;
                    dv{c, 5}=j;
                    dv{c, 6}=ty;
                    dv{c, 7}='V';
                    c=c+1;
                    
                    dv{c, 1}=2;
                    dv{c, 2}=1;
                    dv{c, 3}=field;
                    dv{c, 4}=id;
                    dv{c, 5}=j;
                    dv{c, 6}=ty;
                    dv{c, 7}='N';
                    c=c+1;
              elseif ty==11
                    dv{c, 1}=2;
                    dv{c, 2}=1;
                    dv{c, 3}=field;
                    dv{c, 4}=id;
                    dv{c, 5}=j;
                    dv{c, 6}=ty;
                    dv{c, 7}='u';
                    c=c+1;
                    
                    dv{c, 1}=2;
                    dv{c, 2}=1;
                    dv{c, 3}=field;
                    dv{c, 4}=id;
                    dv{c, 5}=j;
                    dv{c, 6}=ty;
                    dv{c, 7}='v';
                    c=c+1;
                end
            end
            
          end
      end

    end

end
%
% udpate and save back
data.Assembly.Parameter=[data.Assembly.Parameter
                         dv];

