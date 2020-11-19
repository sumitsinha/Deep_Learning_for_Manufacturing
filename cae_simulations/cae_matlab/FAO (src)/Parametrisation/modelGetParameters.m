% Get list of parameters in the model 
function data=modelGetParameters(data)

% Input:
% data: input model

% Output:
% data: udpated data model with updated "data.Assembly.Parameter", with the following entries
    % "Non-ideal part" parameter => non-ideal part definition
                % {1} mode: 0="Non-ideal part" parameter
                % {2} groupID: group ID
                % {3} partID: part ID
                % {4} PointID: ID of control point. Only those controls points with ".Part(partID).Morphing(PointID).Distribution{1}==1/Deterministic" are counted
                % {5} NOT IN USE
                % {6} NOT IN USE
                % {7} Optional
    % "Placement" parameter => rigid part placement parameters
                % {1} mode: 1="Placement" parameter
                % {2} groupID: group ID
                % {3} partID: part ID
                % {4} parameter type: 1/2/3/4/5/6 => alfa, beta, gamma, T, V, N
                % {5} series ID (it may be associated for example to the station ID)
                % {6} reference: 0/1 => global UCS, local UCS to partID
                % {7} Optional
    % "Position" parameter => parameters related to the position of input items(locators, etc.)
                % {1} mode: 2="Position" parameter
                % {2} group ID
                % {3} Field name ('Stitch', 'Hole', 'Slot', 'NcBlock', 'ClampS', 'ClampM', 'CustomConstraint')
                % {4} Field ID
                % {5} Point ID (in case of "stitch", the geometry is defined by 2 points. So "point ID" may be either 1 or 2)
                % {6} parameter type: 2/3/4/5/6/7/8/9/10/11 (T, V, N, TV, TN, VN, TVN, u, v, Disable)
                % {7} parameter name (Reference, T, V,... TVN, u, v, Disable)
%----------------------

% STEP 1: get "Non-ideal part" parameters
data=modelGetParametersNonIdealPart(data);

% STEP 2: get "Placement" parameters
data=modelGetParametersPlacement(data);

% STEP 3: get "Position" parameters
data=modelGetParametersPosition(data);

%----------------------
% Get list of parameters in the model @"Non-ideal part"
function data=modelGetParametersNonIdealPart(data)

% get fields
pData=getInputFieldModel(data, 'Part');

if isempty(pData)
    return
end

% loop over all parts
c=1;
dv=cell(0,7);

npart=length(pData);
for idpart=1:npart
    pgeom=data.Input.Part(idpart).Geometry.Type{1};
    if pgeom==2 % morphing option
        r=length(pData(idpart).Morphing);
        for kr=1:r
            dv{c, 1}=0;
            dv{c, 2}=1;
            dv{c, 3}=idpart;
            dv{c, 4}=kr;
            dv{c, 5}=0;
            dv{c, 6}=0;
            dv{c, 7}='N/A';
            c=c+1;
        end
    end
end

% udpate and save back
data.Assembly.Parameter=[data.Assembly.Parameter
                         dv];     

                     
%----------------------
% Get list of parameters in the model @Placement
function data=modelGetParametersPlacement(data)

% get fields
p=getInputFieldModel(data, 'Part');

if isempty(p)
    return
end

% loop over all parts
c=1;
dv=cell(0,7);
for i=1:length(p)
    pi=p(i);
    ns=size(pi.Parametrisation.Type,1);
    if ns~=length(pi.Parametrisation.UCS);
        error('modelGetParameters (error): Dimensionality mismatch between ".Type" and ".UCS"')
    end
    for k=1:ns
        for j=1:6 % ['alfa', 'beta', 'gamma', 'T', 'V', 'N']
            if pi.Parametrisation.Type(k,j)~=0 % active parameter
                dv{c, 1}=1;
                dv{c, 2}=1;
                dv{c, 3}=i;
                dv{c, 4}=j;
                dv{c, 5}=k;
                dv{c, 6}=pi.Parametrisation.UCS(k);
                dv{c, 7}='N/A';
                c=c+1;
            end
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
            
            if ty>1 && ~isempty(ty) % parametric
                if ty==2 || ty==3 || ty==4 || ty==9 || ty==10 || ty==12
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
                    
               elseif ty==13
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
                    dv{c, 7}='ON/OFF';
                    c=c+1;
                                        
               elseif ty==14
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
                    dv{c, 7}='ON/OFF';
                    c=c+1;
                                        
               elseif ty==15
                    dv{c, 1}=2;
                    dv{c, 2}=1;
                    dv{c, 3}=field;
                    dv{c, 4}=id;
                    dv{c, 5}=j;
                    dv{c, 6}=ty;
                    dv{c, 7}='N';
                    c=c+1;
                    
                    dv{c, 1}=2;
                    dv{c, 2}=1;
                    dv{c, 3}=field;
                    dv{c, 4}=id;
                    dv{c, 5}=j;
                    dv{c, 6}=ty;
                    dv{c, 7}='ON/OFF';
                    c=c+1;
                                        
               elseif ty==16
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
                    dv{c, 7}='ON/OFF';
                    c=c+1;
                                                            
               elseif ty==17
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
                    
                    dv{c, 1}=2;
                    dv{c, 2}=1;
                    dv{c, 3}=field;
                    dv{c, 4}=id;
                    dv{c, 5}=j;
                    dv{c, 6}=ty;
                    dv{c, 7}='ON/OFF';
                    c=c+1;
                                                            
               elseif ty==18
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
                    
                    dv{c, 1}=2;
                    dv{c, 2}=1;
                    dv{c, 3}=field;
                    dv{c, 4}=id;
                    dv{c, 5}=j;
                    dv{c, 6}=ty;
                    dv{c, 7}='ON/OFF';
                    c=c+1;
                                                            
               elseif ty==19
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
                    
                    dv{c, 1}=2;
                    dv{c, 2}=1;
                    dv{c, 3}=field;
                    dv{c, 4}=id;
                    dv{c, 5}=j;
                    dv{c, 6}=ty;
                    dv{c, 7}='ON/OFF';
                    c=c+1;
                                                                                
               elseif ty==20
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
                    dv{c, 7}='ON/OFF';
                    c=c+1;
                                                                                 
               elseif ty==21
                    dv{c, 1}=2;
                    dv{c, 2}=1;
                    dv{c, 3}=field;
                    dv{c, 4}=id;
                    dv{c, 5}=j;
                    dv{c, 6}=ty;
                    dv{c, 7}='v';
                    c=c+1;
                    
                    dv{c, 1}=2;
                    dv{c, 2}=1;
                    dv{c, 3}=field;
                    dv{c, 4}=id;
                    dv{c, 5}=j;
                    dv{c, 6}=ty;
                    dv{c, 7}='ON/OFF';
                    c=c+1;
                                                                                
               elseif ty==22
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
                                        
                    dv{c, 1}=2;
                    dv{c, 2}=1;
                    dv{c, 3}=field;
                    dv{c, 4}=id;
                    dv{c, 5}=j;
                    dv{c, 6}=ty;
                    dv{c, 7}='ON/OFF';
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

