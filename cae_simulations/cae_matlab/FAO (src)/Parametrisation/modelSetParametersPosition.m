% Set parameter to the model
function data=modelSetParametersPosition(data, X, field, id, idpoint, ptype, pname)

% Inputs
% data: data structure
% X: sampled value
% field: Field name ('Stitch', 'Hole', 'Slot', 'NcBlock', 'ClampS', 'ClampM', 'CustomConstraint')
% id: Field ID
% idpoint: Point ID (in case of "stitch", the geometry is defined by 2 points. So "point ID" may be either 1 or 2)
% ptype: parameter type: 2/3/4/5/6/7/8/9/10 (T, V, N, TV, TN, VN, TVN, u, v)
% pname: parameter name (Reference, T, V,... TVN, u, v)

% Outputs
% data: updated data structure 

%--
[fid, ~]=retrieveStructure(data, field, id); 

if ptype>1 % parametric
    if ptype==2 || ptype==3 || ptype==4 || ptype==9 || ptype==10
        if ptype==2
            fid.Parametrisation.Geometry.T{idpoint}=X;
        elseif ptype==3
            fid.Parametrisation.Geometry.V{idpoint}=X;
        elseif ptype==4
            fid.Parametrisation.Geometry.N{idpoint}=X;
        elseif ptype==9
            fid.Parametrisation.DoC.u{idpoint}=X;
        elseif ptype==10
            fid.Parametrisation.DoC.v{idpoint}=X;
        else
            flag=false;
        end
    elseif ptype==5
       if strcmp(pname, 'T')
           fid.Parametrisation.Geometry.T{idpoint}=X;
       elseif strcmp(pname, 'V')
           fid.Parametrisation.Geometry.V{idpoint}=X;
       else
           flag=false;
       end
    elseif ptype==6
       if strcmp(pname, 'T')
           fid.Parametrisation.Geometry.T{idpoint}=X;
       elseif strcmp(pname, 'N')
           fid.Parametrisation.Geometry.N{idpoint}=X;
       else
           flag=false;
       end
    elseif ptype==7
       if strcmp(pname, 'V')
           fid.Parametrisation.Geometry.V{idpoint}=X;
       elseif strcmp(pname, 'N')
           fid.Parametrisation.Geometry.N{idpoint}=X;
       else
           flag=false;
       end
    elseif ptype==8
       if strcmp(pname, 'T')
           fid.Parametrisation.Geometry.T{idpoint}=X;
       elseif strcmp(pname, 'V')
           fid.Parametrisation.Geometry.V{idpoint}=X;
       elseif strcmp(pname, 'N')
           fid.Parametrisation.Geometry.N{idpoint}=X;
       else
           flag=false;
       end
    elseif ptype==11
       if strcmp(pname, 'u')
            fid.Parametrisation.DoC.u{idpoint}=X;
       elseif strcmp(pname, 'v')
            fid.Parametrisation.DoC.v{idpoint}=X;
       end
    else
        flag=false;
    end
else
    flag=false;
end

% save back
data=retrieveBackStructure(data, fid, field, id);


