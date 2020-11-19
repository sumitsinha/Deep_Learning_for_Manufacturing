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
    % Only continous variables
    if ptype==2 || ptype==3 || ptype==4 || ptype==9 || ptype==10 || ptype==12
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
        elseif ptype==12
            fid.Enable=logical(X);
            fid.EnableReset=logical(X);
        end
    elseif ptype==5
       if strcmp(pname, 'T')
           fid.Parametrisation.Geometry.T{idpoint}=X;
       elseif strcmp(pname, 'V')
           fid.Parametrisation.Geometry.V{idpoint}=X;
       end
    elseif ptype==6
       if strcmp(pname, 'T')
           fid.Parametrisation.Geometry.T{idpoint}=X;
       elseif strcmp(pname, 'N')
           fid.Parametrisation.Geometry.N{idpoint}=X;
       end
    elseif ptype==7
       if strcmp(pname, 'V')
           fid.Parametrisation.Geometry.V{idpoint}=X;
       elseif strcmp(pname, 'N')
           fid.Parametrisation.Geometry.N{idpoint}=X;
       end
    elseif ptype==8
       if strcmp(pname, 'T')
           fid.Parametrisation.Geometry.T{idpoint}=X;
       elseif strcmp(pname, 'V')
           fid.Parametrisation.Geometry.V{idpoint}=X;
       elseif strcmp(pname, 'N')
           fid.Parametrisation.Geometry.N{idpoint}=X; 
       end
    elseif ptype==11
       if strcmp(pname, 'u')
            fid.Parametrisation.DoC.u{idpoint}=X;
       elseif strcmp(pname, 'v')
            fid.Parametrisation.DoC.v{idpoint}=X;
       end
    end
    %--------------- ...continous and binary variables
    if ptype==13 || ptype==14 || ptype==15 || ptype==20 || ptype==21
        if strcmp(pname, 'ON/OFF')
            fid.Enable=logical(X);
            fid.EnableReset=logical(X);
        else
            if ptype==13
                fid.Parametrisation.Geometry.T{idpoint}=X;
            elseif ptype==14
                fid.Parametrisation.Geometry.V{idpoint}=X;
            elseif ptype==15
                fid.Parametrisation.Geometry.N{idpoint}=X;
            elseif ptype==20
                fid.Parametrisation.DoC.u{idpoint}=X;
            elseif ptype==21
                fid.Parametrisation.DoC.v{idpoint}=X;
            end
        end
    elseif ptype==16
       if strcmp(pname, 'ON/OFF')
            fid.Enable=logical(X);
            fid.EnableReset=logical(X);
       else
           if strcmp(pname, 'T')
               fid.Parametrisation.Geometry.T{idpoint}=X;
           elseif strcmp(pname, 'V')
               fid.Parametrisation.Geometry.V{idpoint}=X;
           end
       end
    elseif ptype==17
       if strcmp(pname, 'ON/OFF')
            fid.Enable=logical(X);
            fid.EnableReset=logical(X);
       else
           if strcmp(pname, 'T')
               fid.Parametrisation.Geometry.T{idpoint}=X;
           elseif strcmp(pname, 'N')
               fid.Parametrisation.Geometry.N{idpoint}=X;
           end
       end
    elseif ptype==18
       if strcmp(pname, 'ON/OFF')
            fid.Enable=logical(X);
            fid.EnableReset=logical(X);
       else
           if strcmp(pname, 'V')
               fid.Parametrisation.Geometry.V{idpoint}=X;
           elseif strcmp(pname, 'N')
               fid.Parametrisation.Geometry.N{idpoint}=X;
           end
       end
    elseif ptype==19
       if strcmp(pname, 'ON/OFF')
            fid.Enable=logical(X);
            fid.EnableReset=logical(X);
       else
           if strcmp(pname, 'T')
               fid.Parametrisation.Geometry.T{idpoint}=X;
           elseif strcmp(pname, 'V')
               fid.Parametrisation.Geometry.V{idpoint}=X;
           elseif strcmp(pname, 'N')
               fid.Parametrisation.Geometry.N{idpoint}=X; 
           end
       end
    elseif ptype==22
       if strcmp(pname, 'ON/OFF')
            fid.Enable=logical(X);
            fid.EnableReset=logical(X);
       else
           if strcmp(pname, 'u')
                fid.Parametrisation.DoC.u{idpoint}=X;
           elseif strcmp(pname, 'v')
                fid.Parametrisation.DoC.v{idpoint}=X;
           end
       end
    end
end

% save back
data=retrieveBackStructure(data, fid, field, id);


