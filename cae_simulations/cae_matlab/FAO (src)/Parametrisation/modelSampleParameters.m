% Sample design parameters
function data=modelSampleParameters(data)

% Inputs:
% data: data structure with the following inputs:
    % .Assembly.SamplingStrategy{1}=idS: option for sampling
        % idS=1 => full factorial: full factorial design
        % idS=2 => random: random sampling
        % idS=3 => user: user table
    % .Assembly.SamplingOptions
        % .SampleSize=sample size (only for SamplingStrategy="random")
        % .IdTable=id of the parameter table (only for SamplingStrategy="user")
% .Assembly.Parameter: design parameter (CELL matrix) as follows
    % "Non-ideal part" parameter => non-ideal part definition
            % {1} mode: 0="Non-ideal part" parameter
            % {2} groupID: group ID
            % {3} partID: part ID
            % {4} PointID: ID of control point. Only those controls points with ".Part(partID).Morphing(PointID).Distribution{1}==1/Deterministic" are counted
            % {5} NOT IN USE
            % {6} NOT IN USE
            % {7} NOT IN USE
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
% .Assembly.Group: group variables (NUMERIC matrix)
    % [groupID, min, max, resolution]

%-------  
% Outputs:
% data: updated data structure with the following fields:
    % .Assembly.X.Value: [n. samples x no. of parameters]
    % .Assembly.X.nD: of parameters
    % .Assembly.X.Status: [n. samples x no. of parameters] =>0/1 not solved / solved

%-------
dv=data.Assembly.Parameter;
if isempty(dv)
    warning('Sampling design parameter (warning): no parameter defined!\n');
    return
end
%
dg=data.Assembly.Group;
if isempty(dg)
    warning('Sampling design parameter (warning): no group defined!\n');
    return
end

%------------
% find no. of variables to be allocated 
grouplist=unique(cell2mat(dv(:,2)));
nvars=length(grouplist); % no. independent parameters

% find no. of levels to be allocated to the table
mlevels=dg(grouplist,2); % min
Mlevels=dg(grouplist,3); % max
nlevels=dg(grouplist,4); % resolution

%--------------
Xtemp=modelGenerateSamples(data, mlevels, Mlevels, nlevels);
%--------------

% STEP 2: assign to the whole dependent variables
npara=size(dv,1); % no. dependent parameters
    % Xout=zeros(size(Xtemp,1),npara);
Xout=data.Assembly.X.Value;
for i=1:npara
   temp=0;
   for j=1:nvars
        if dv{i,2}==grouplist(j)
            temp=j;
            break
        end
   end
   %--
   if temp==0
       Xout(:,i)=Xtemp(:,i)*Xout(:,i);
   else
       Xout(:,i)=Xtemp(:, temp).*Xout(:,i);
   end
end
%
% Save back
data.Assembly.X.Value=Xout;
data.Assembly.X.nD=size(Xout,1);
data.Assembly.X.Status=zeros(data.Assembly.X.nD,1);
