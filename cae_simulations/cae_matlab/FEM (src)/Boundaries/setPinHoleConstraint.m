% set pin-hole constraint for "thin shell" elements
function fem=setPinHoleConstraint(fem)

% For thin sheet metals rotation around X and Y can be assumed free 

nph=length(fem.Boundary.Constraint.PinHole);

if nph==0
    return
end

for i=1:nph
    
    %---
    Pm=fem.Boundary.Constraint.PinHole(i).Pm;
    Nm1=fem.Boundary.Constraint.PinHole(i).Nm1;
    Nm2=fem.Boundary.Constraint.PinHole(i).Nm2;
    
    iddom=fem.Boundary.Constraint.PinHole(i).Domain;
    dsear=fem.Boundary.Constraint.PinHole(i).SearchDist;
    
    value=fem.Boundary.Constraint.PinHole(i).Value;
        
    % apply constraint along Nm1 and Nm2

    nc=length(fem.Boundary.Constraint.Bilateral.Element)+1;

    % along Nx
    fem.Boundary.Constraint.Bilateral.Element(nc).Pm=Pm;
    fem.Boundary.Constraint.Bilateral.Element(nc).Reference='vectorTra';
    fem.Boundary.Constraint.Bilateral.Element(nc).Nm=Nm1;
    fem.Boundary.Constraint.Bilateral.Element(nc).Value=value(1);
    fem.Boundary.Constraint.Bilateral.Element(nc).Domain=iddom;
    fem.Boundary.Constraint.Bilateral.Element(nc).SearchDist=dsear;
 
    % along Ny
    nc=nc+1;
    fem.Boundary.Constraint.Bilateral.Element(nc).Pm=Pm;
    fem.Boundary.Constraint.Bilateral.Element(nc).Reference='vectorTra';
    fem.Boundary.Constraint.Bilateral.Element(nc).Nm=Nm2;
    fem.Boundary.Constraint.Bilateral.Element(nc).Value=value(2);
    fem.Boundary.Constraint.Bilateral.Element(nc).Domain=iddom;
    fem.Boundary.Constraint.Bilateral.Element(nc).SearchDist=dsear;
 
    % set references for low-level constraint
    fem.Boundary.Constraint.PinHole(i).ConstraintId=[nc-1 nc];

end

                    %     % around Nx
                    %     nc=nc+1;
                    %     fem.Boundary.Constraint.Bilateral.Element(nc).Pm=Pm;
                    %     fem.Boundary.Constraint.Bilateral.Element(nc).Reference='vectorRot';
                    %     fem.Boundary.Constraint.Bilateral.Element(nc).Nm=Nm1;
                    %     fem.Boundary.Constraint.Bilateral.Element(nc).Value=value(3);
                    %     fem.Boundary.Constraint.Bilateral.Element(nc).Domain=iddom;
                    %     fem.Boundary.Constraint.Bilateral.Element(nc).SearchDist=dsear;
                    % 
                    %     % around Ny
                    %     nc=nc+1;
                    %     fem.Boundary.Constraint.Bilateral.Element(nc).Pm=Pm;
                    %     fem.Boundary.Constraint.Bilateral.Element(nc).Reference='vectorRot';
                    %     fem.Boundary.Constraint.Bilateral.Element(nc).Nm=Nm2;
                    %     fem.Boundary.Constraint.Bilateral.Element(nc).Value=value(4);
                    %     fem.Boundary.Constraint.Bilateral.Element(nc).Domain=iddom;
                    %     fem.Boundary.Constraint.Bilateral.Element(nc).SearchDist=dsear;
                    
                    %     fem.Boundary.Constraint.PinHole(i).ConstraintId=[nc-3 nc-2 nc-1 nc];
