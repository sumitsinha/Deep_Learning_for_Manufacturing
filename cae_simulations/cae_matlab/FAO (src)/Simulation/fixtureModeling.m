% Set fixture data
function fem=fixtureModeling(data, fem, geomparaid)

% data: data structure
% geomparaid: parameter id
% fem: updated fem model structure with boundary conditions

%--------
eps=1e-6;
%--------

%--------------------------------------------------------------------------
% initialise counters
fem.Boundary.Constraint.Bilateral.Node=[];
fem.Boundary.Constraint.Bilateral.Element=[];

fem.Boundary.Constraint.RigidLink=[];

fem.Boundary.Constraint.PinHole=[];
fem.Boundary.Constraint.PinSlot=[];

fem.Boundary.Constraint.Unilateral=[];

fem.Boundary.DimplePair=[];
fem.Boundary.ContactPair=[];

fem.Boundary.Load.Node=[];
fem.Boundary.Load.User=[];
%-----
% Initialise counters
%
countBLN=1; % bilateral node
countBLE=1; % bilateral element
countUL=1; % unilateral
%
% define rigid links
field='Stitch';
f=getInputFieldModel(data, field);
n=length(f);
count=1;
for i=1:n

    % get field
    fi=f(i);

    if fi.Type{1}==3 % rigid link
        % check status
        [flag, geomparaidi]=checkInputStatusGivenPoint(fi, geomparaid, 1);

        if flag
            Nm=fi.Nam{1}(geomparaidi,:);
            L=norm(Nm);

            if L>=eps
                Nm=Nm/L;
                
%                 part_to_part_gap=norm(fi.Pam{1}(geomparaidi,:)-fi.Pas{1}(geomparaidi,:));
%                 if part_to_part_gap <= fi.Gap
                    fem.Boundary.Constraint.RigidLink(count).Pm=fi.Pam{1}(geomparaidi,:); 
                    fem.Boundary.Constraint.RigidLink(count).Nm=Nm;
                    fem.Boundary.Constraint.RigidLink(count).SearchDist=fi.SearchDist(1);
                    fem.Boundary.Constraint.RigidLink(count).Master=fi.DomainM;
                    fem.Boundary.Constraint.RigidLink(count).Slave=fi.DomainS;

                    fem.Boundary.Constraint.RigidLink(count).Frame='ref';

                    count=count+1;
%                 end
            end

        end
    end
    
end
          
% define rigid or flexible option 
field='Part';
f=getInputFieldModel(data, field);
n=length(f);        
for i=1:n

    if f(i).Status==0 && f(i).Enable

        fg=f(i).FlexStatus;

        if ~fg % rigid mode
            idnode = fem.Domain(i).Node;

            fem.Boundary.Constraint.Bilateral.Node(countBLN).Node=idnode;
            fem.Boundary.Constraint.Bilateral.Node(countBLN).Reference='cartesian';
            fem.Boundary.Constraint.Bilateral.Node(countBLN).DoF=[1 2 3 4 5 6]; % rigid
            fem.Boundary.Constraint.Bilateral.Node(countBLN).Value=[0 0 0 0 0 0];

            countBLN=countBLN+1;
        end
    end
end

%--------------------------------------------------------------------------
% define sub-model constraints and pre-loads
if isfield(data, 'Assembly')
    % Sub-models
    if data.Assembly.Solver.UseSubModel 
        if ~isempty(data.Assembly.SubModel.Node)
            nNode=length(data.Assembly.SubModel.Node);
            for i=1:nNode
                fem.Boundary.Constraint.Bilateral.Node(countBLN).Node=data.Assembly.SubModel.Node(i);
                fem.Boundary.Constraint.Bilateral.Node(countBLN).DoF=data.Assembly.SubModel.DoF; 
                fem.Boundary.Constraint.Bilateral.Node(countBLN).Value=data.Assembly.SubModel.U(i,:);
                fem.Boundary.Constraint.Bilateral.Node(countBLN).Reference='cartesian';
                
                countBLN=countBLN+1;
            end
        end
    end
    %
    % Pre-loads
    if data.Assembly.Solver.UsePreLoad % use pre-load
        if ~isempty(data.Assembly.PreLoad.F)
            fem.Boundary.Load.User.DofId=data.Assembly.PreLoad.DoF;
            fem.Boundary.Load.User.Value=data.Assembly.PreLoad.F;
        end
    end
    
end

%--------------------------------------------------------------------------
% define pin-hole settings
field='Hole';
f=getInputFieldModel(data, field);
n=length(f);

count=1;
for i=1:n

    % get field
    fi=f(i);

    % check status
    [flag, geomparaidi]=checkInputStatusGivenPoint(fi, geomparaid, 1);

    if flag
        % calculate rotation matrix      
        Rc=f(i).Parametrisation.Geometry.R{1};  
%         
%         value=[f.Parametrisation.DoC.u{1}(geomparaidi),...
%                f.Parametrisation.DoC.v{1}(geomparaidi)];
        value=[0 0];
        
        fem.Boundary.Constraint.PinHole(count).Pm=fi.Pam{1}(geomparaidi,:); 
        fem.Boundary.Constraint.PinHole(count).Nm1=Rc(:,1)'; 
        fem.Boundary.Constraint.PinHole(count).Nm2=Rc(:,2)'; 
        fem.Boundary.Constraint.PinHole(count).Domain=fi.DomainM; 
        fem.Boundary.Constraint.PinHole(count).SearchDist=fi.SearchDist(1); 
        fem.Boundary.Constraint.PinHole(count).Value=value;

        count=count+1;
    end

end
    

%--------------------------------------------------------------------------
% define pin-slot settings
field='Slot';
f=getInputFieldModel(data, field);
n=length(f);

count=1;
for i=1:n

    % get field
    fi=f(i);

    % check status
    [flag, geomparaidi]=checkInputStatusGivenPoint(fi, geomparaid, 1);

    if flag
        
        % calculate rotation matrix       
        Rc=f(i).Parametrisation.Geometry.R{1};  
        
%         value=f.Parametrisation.DoC.u{1}(geomparaidi);
        value=0;
        
        fem.Boundary.Constraint.PinSlot(count).Pm=fi.Pam{1}(geomparaidi,:); 
        fem.Boundary.Constraint.PinSlot(count).Nm1=Rc(:,1)'; 
        fem.Boundary.Constraint.PinSlot(count).Domain=fi.DomainM; 
        fem.Boundary.Constraint.PinSlot(count).SearchDist=fi.SearchDist(1); 
        fem.Boundary.Constraint.PinSlot(count).Value=value;

        count=count+1;
    end

end


%--------------------------------------------------------------------------
% define custom constraints    
field='CustomConstraint';
f=getInputFieldModel(data, field);
n=length(f);
for i=1:n

    % get field
    fi=f(i);

    % check status
    [flag, geomparaidi]=checkInputStatusGivenPoint(fi, geomparaid, 1);

    if flag

        typei=fi.Type{1}; % type
        if typei==1 % cartesian
            tdofs=fi.DoFs;
            tvalues=fi.Value;
            dofs=[];
            values=[];
            for j=1:6
                if tdofs(j)
                    dofs=[dofs j]; %#ok<AGROW>
                    values=[values tvalues(j)]; %#ok<AGROW>
                end
            end

            fem.Boundary.Constraint.Bilateral.Element(countBLE).Pm=fi.Pam{1}(geomparaidi,:); 
            fem.Boundary.Constraint.Bilateral.Element(countBLE).Reference='cartesian'; 
            fem.Boundary.Constraint.Bilateral.Element(countBLE).SearchDist=fi.SearchDist(1); 
            fem.Boundary.Constraint.Bilateral.Element(countBLE).DoF=dofs;
            fem.Boundary.Constraint.Bilateral.Element(countBLE).Value=values;
            fem.Boundary.Constraint.Bilateral.Element(countBLE).Domain=fi.DomainM; 

            countBLE=countBLE+1;
        elseif typei==2 % vectorTra
            fem.Boundary.Constraint.Bilateral.Element(countBLE).Pm=fi.Pam{1}(geomparaidi,:); 
            fem.Boundary.Constraint.Bilateral.Element(countBLE).Reference='vectorTra'; 
            fem.Boundary.Constraint.Bilateral.Element(countBLE).SearchDist=fi.SearchDist(1); 
            fem.Boundary.Constraint.Bilateral.Element(countBLE).Value=fi.Value;
            fem.Boundary.Constraint.Bilateral.Element(countBLE).Nm=fi.Nam{1}(geomparaidi,:);
            fem.Boundary.Constraint.Bilateral.Element(countBLE).Domain=fi.DomainM; 

            countBLE=countBLE+1;
        elseif typei==3 % unilateralLock
            fem.Boundary.Constraint.Unilateral(countUL).Pm=fi.Pam{1}(geomparaidi,:); 
            fem.Boundary.Constraint.Unilateral(countUL).SearchDist=fi.SearchDist(1); 
            fem.Boundary.Constraint.Unilateral(countUL).Nm=fi.Nam{1}(geomparaidi,:); 
            fem.Boundary.Constraint.Unilateral(countUL).Size=false; 
            fem.Boundary.Constraint.Unilateral(countUL).Offset=0.0;
            fem.Boundary.Constraint.Unilateral(countUL).Domain=fi.DomainM; 
            fem.Boundary.Constraint.Unilateral(countUL).Constraint='lock'; % use lock option
            fem.Boundary.Constraint.Unilateral(countUL).Frame='ref';

            countUL=countUL+1;
        elseif typei==4 % unilateralFree
            fem.Boundary.Constraint.Unilateral(countUL).Pm=fi.Pam{1}(geomparaidi,:); 
            fem.Boundary.Constraint.Unilateral(countUL).SearchDist=fi.SearchDist(1); 
            fem.Boundary.Constraint.Unilateral(countUL).Nm=fi.Nam{1}(geomparaidi,:); 
            fem.Boundary.Constraint.Unilateral(countUL).Size=false; 
            fem.Boundary.Constraint.Unilateral(countUL).Offset=0.0;
            fem.Boundary.Constraint.Unilateral(countUL).Domain=fi.DomainM; 
            fem.Boundary.Constraint.Unilateral(countUL).Constraint='free'; % use free option
            fem.Boundary.Constraint.Unilateral(countUL).Frame='ref';

            countUL=countUL+1;
        end

    end
            
end 


%--------------------------------------------------------------------------
% define locators: NC-Block
field='NcBlock';
f=getInputFieldModel(data, field);
n=length(f);
for i=1:n

    % get field
    fi=f(i);

    np=length(fi.Status); % number of points

    for j=1:np

        % check status
        [flag, geomparaidi]=checkInputStatusGivenPoint(fi, geomparaid, j);

        if flag

            fem.Boundary.Constraint.Unilateral(countUL).Pm=fi.Pam{j}(geomparaidi,:); 
            fem.Boundary.Constraint.Unilateral(countUL).SearchDist=fi.SearchDist(1); 
            fem.Boundary.Constraint.Unilateral(countUL).Nm=fi.Nam{j}(geomparaidi,:); 
            fem.Boundary.Constraint.Unilateral(countUL).Size=false; 
            fem.Boundary.Constraint.Unilateral(countUL).Offset=0.0;
            fem.Boundary.Constraint.Unilateral(countUL).Domain=fi.DomainM; 
            fem.Boundary.Constraint.Unilateral(countUL).Constraint='free'; % use free option
            fem.Boundary.Constraint.Unilateral(countUL).Frame='ref';

            countUL=countUL+1;
        end

    end

end



% define locators: ClampS
field='ClampS';
f=getInputFieldModel(data, field);
n=length(f);
for i=1:n

    % get field
    fi=f(i);

    np=length(fi.Status); % number of points

    for j=1:np

        % check status
        [flag, geomparaidi]=checkInputStatusGivenPoint(fi, geomparaid, j);
        
        if flag

            fem.Boundary.Constraint.Unilateral(countUL).Pm=fi.Pam{j}(geomparaidi,:); 
            fem.Boundary.Constraint.Unilateral(countUL).SearchDist=fi.SearchDist(1); 
            fem.Boundary.Constraint.Unilateral(countUL).Nm=fi.Nam{j}(geomparaidi,:); 
            fem.Boundary.Constraint.Unilateral(countUL).Size=false; 
            fem.Boundary.Constraint.Unilateral(countUL).Offset=0.0;
            fem.Boundary.Constraint.Unilateral(countUL).Domain=fi.DomainM; 
            fem.Boundary.Constraint.Unilateral(countUL).Constraint='lock'; % use lock option
            fem.Boundary.Constraint.Unilateral(countUL).Frame='ref';

            countUL=countUL+1;
        end
    end
    
end


% define locators: ClampM
field='ClampM';
f=getInputFieldModel(data, field);
n=length(f);
for i=1:n

    % get field
    fi=f(i);

    np=length(fi.Status); % number of points

    for j=1:np

        % check status
        [flag, geomparaidi]=checkInputStatusGivenPoint(fi, geomparaid, j);

        if flag

                % master
                fem.Boundary.Constraint.Unilateral(countUL).Pm=fi.Pam{j}(geomparaidi,:); 
                fem.Boundary.Constraint.Unilateral(countUL).SearchDist=fi.SearchDist(1); 
                fem.Boundary.Constraint.Unilateral(countUL).Nm=fi.Nam{j}(geomparaidi,:); 
                fem.Boundary.Constraint.Unilateral(countUL).Size=false; 
                fem.Boundary.Constraint.Unilateral(countUL).Offset=0.0;
                fem.Boundary.Constraint.Unilateral(countUL).Domain=fi.DomainM; 
                fem.Boundary.Constraint.Unilateral(countUL).Constraint='free'; 
                fem.Boundary.Constraint.Unilateral(countUL).Frame='ref';

                countUL=countUL+1;

                % slave
                fem.Boundary.Constraint.Unilateral(countUL).Pm=fi.Pas{j}(geomparaidi,:); 
                fem.Boundary.Constraint.Unilateral(countUL).SearchDist=fi.SearchDist(1); 
                fem.Boundary.Constraint.Unilateral(countUL).Nm=fi.Nas{j}(geomparaidi,:); 
                fem.Boundary.Constraint.Unilateral(countUL).Size=false; 
                fem.Boundary.Constraint.Unilateral(countUL).Offset=0.0;
                fem.Boundary.Constraint.Unilateral(countUL).Domain=fi.DomainS; 
                fem.Boundary.Constraint.Unilateral(countUL).Constraint='free'; 
                fem.Boundary.Constraint.Unilateral(countUL).Frame='ref';

                countUL=countUL+1;
        end
    end
end

% define dimples
field='Dimple';
f=getInputFieldModel(data, field);
n=length(f);
    
count=1;
for i=1:n 
    
    % get field
    fi=f(i);

    [flag, geomparaidi]=checkInputStatusGivenPoint(fi, geomparaid, 1);

    if flag

        fem.Boundary.DimplePair(count).Pm=fi.Pam{1}(geomparaidi,:); 
        fem.Boundary.DimplePair(count).Master=fi.DomainM;
        if fi.MasterFlip
             fem.Boundary.DimplePair(count).MasterFlip=true;
        else
             fem.Boundary.DimplePair(count).MasterFlip=false;
        end
        fem.Boundary.DimplePair(count).Slave=fi.DomainS;
        fem.Boundary.DimplePair(count).SearchDist=fi.SearchDist(1);
        fem.Boundary.DimplePair(count).Height=fi.Height;
        fem.Boundary.DimplePair(count).Offset=0.0;
        fem.Boundary.DimplePair(count).Frame='ref';
        
        count=count+1;
    end
end


% define contact pair
field='Contact';
f=getInputFieldModel(data, field);
n=length(f);  

count=1;
for i=1:n

    fi=f(i);
    
    if fi.Status{1}==0 && fi.Enable  
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

