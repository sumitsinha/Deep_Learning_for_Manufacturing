% Import model input data from external "filepath"
function data=modelImportInput(data, filepath, flag, formatFile)

% data: data input
% filepath: file location
% flag:
    % "Stitch" => stitch layout
        % [type, master, slave, domain(m), domain(s), Ps, Pe, Pv]
            % if type==1 (linear) => Pv=[0,0,0] (not used)
            % if type==2(circular) or 3 (rigid link) => Pe=Pv=[0,0,0] (not used)
            % if type==4 (edge) => Ps, Pe, Pv are mandatory inputs
    % "Hole" / "Slot" => hole/slot layout
        % [master, domain(m), Ps, Nm, Nt]
            % Nm: normal vector
            % Nt: tangent vector
            % if "Hole" => Nt=any
            % if "Slot" => Nt is the axis of constraint of the slot
    % "ClampM" => clampM layout
        % [master, slave domain(m), domain(s) P]
             % P: position of the locator
    % "ClampS" => clampS layout
        % [master, slave domain(m), domain(s) P]
            % P: position of the locator
    % "NcBlock" => NcBlock layout
        % [master, domain(m), P]
            % P: position of the locator
    % "CustomConstraint" => user defined constraints layout
        % [master, slave domain(m), domain(s) P]
            % P: position of the locator
    % "Contact" => contact pairs
        % [master, slave]
    % "Morphing" => input for morphing mesh
        % [master, Pc]
            % Pc: position of control point
% formatFile: format file for additional input data  => cell array with the following fields
    % Nm: [1x3] => Normal vector
    % Nt: [1x3] => Tangent vector
    %..
    
% Note
    % Master/Slave => used for projection on the geometry to calculate reference positions
    % DomainM/DomainS => used for calculation by FEM kernel
%------------------------

if nargin
    formatFile={''};
end

% check format
if strcmp(flag,'Stitch')
    maxcol=14; % [type, master, slave, domain(m), domain(s), Ps, Pe, Pv]
    % if type==1 (linear) => Pv=[0,0,0] (not used)
    % if type==2(circular) or 3 (rigid link) => Pe=Pv=[0,0,0] (not used)
    % if type==4 (edge) => Ps, Pe, Pv are mandatory inputs
elseif strcmp(flag,'Hole') || strcmp(flag,'Slot')
    maxcol=11; % [master, domain(m), Ps, Nm, Nt]
    % Nm: normal vector
    % Nt: tangent vector
    % if "Hole" => Nt=any
    % if "Slot" => Nt is the axis of constraint of the slot
elseif strcmp(flag,'ClampM') 
    maxcol=7; % [master, slave domain(m), domain(s) P]
    % P: position of the locator
elseif strcmp(flag,'NcBlock') 
    maxcol=5; % [master, domain(m), P]
    % P: position of the locator
elseif strcmp(flag,'Contact') 
    maxcol=2; % [master, slave]
elseif strcmp(flag,'Morphing') 
    maxcol=4; % [master, Pc]
    % Pc: position of control point
else
    maxcol=5; % [master, domain(m), P]
    % P: position of the input
end
%--
[maxcol, colPos]=format2Columns(maxcol, formatFile);
%--
% import data
d=modelLoadInputFile(filepath, maxcol, true);
%--
% update database
if ~isempty(d)
    nd=size(d,1);
    for i=1:nd
        
        % add new item
        if strcmp(flag, 'Morphing')
            if isfield(data.Input, 'Part')
                nP=length(data.Input.Part);
                if d(i,1)>0 && d(i,1)<=nP
                    [fp,flagp]=retrieveStructure(data, 'Part', d(i,1));
                    
                    if flagp
                        % initialise new fields
                        fp.Morphing(i)=initMorphingMesh();
                        
                        %... and update
                        fp.Morphing(i).Pc=d(i,[2 3 4]);
                                                
                        % add new selection
                        data=modelAddItem(data, 'Selection');
                        cSelection=length(data.Input.Selection);
                        
                        data.Input.Selection(cSelection).Pm=d(i,[2 3 4]);
                        data.Input.Selection(cSelection).PmReset=d(i,[2 3 4]);
                        fp.Morphing(i).Selection=cSelection;
                        
                        % save back
                        data=retrieveBackStructure(data, fp, 'Part', d(i,1));
                    end
                end
            end
        else
            data=modelAddItem(data, flag);
        end
        
        if strcmp(flag,'Stitch')
            data.Input.Stitch(end).Type{1}=d(i,1);
            data.Input.Stitch(end).Master=d(i,2);
            data.Input.Stitch(end).Slave=d(i,3);
            data.Input.Stitch(end).DomainM=d(i,4);
            data.Input.Stitch(end).DomainS=d(i,5);
            data.Input.Stitch(end).Pm(1,:)=d(i,[6 7 8]);
            data.Input.Stitch(end).Pm(2,:)=d(i,[9 10 11]);
            data.Input.Stitch(end).Pm(3,:)=d(i,[12 13 14]);

            data.Input.Stitch(end).PmReset=data.Input.Stitch(end).Pm;

        elseif strcmp(flag,'Hole')

            data.Input.PinLayout.Hole(end).Master=d(i,1);
            data.Input.PinLayout.Hole(end).DomainM=d(i,2);
            data.Input.PinLayout.Hole(end).Pm=d(i,[3 4 5]);
            data.Input.PinLayout.Hole(end).PmReset=data.Input.PinLayout.Hole(end).Pm;
            data.Input.PinLayout.Hole(end).Nm=d(i,[6 7 8]);
            data.Input.PinLayout.Hole(end).NmReset=data.Input.PinLayout.Hole(end).Nm;
            data.Input.PinLayout.Hole(end).TangentType{1}=2; % model
            data.Input.PinLayout.Hole(end).Nt=d(i,[9 10 11]);
            data.Input.PinLayout.Hole(end).NtReset=data.Input.PinLayout.Hole(end).Nt;

        elseif strcmp(flag,'Slot')

            data.Input.PinLayout.Slot(end).Master=d(i,1);
            data.Input.PinLayout.Slot(end).DomainM=d(i,2);
            data.Input.PinLayout.Slot(end).Pm=d(i,[3 4 5]);
            data.Input.PinLayout.Slot(end).PmReset=data.Input.PinLayout.Slot(end).Pm;
            data.Input.PinLayout.Slot(end).Nm=d(i,[6 7 8]);
            data.Input.PinLayout.Slot(end).NmReset=data.Input.PinLayout.Slot(end).Nm;
            data.Input.PinLayout.Slot(end).Nt=d(i,[9 10 11]);
            data.Input.PinLayout.Slot(end).NtReset=data.Input.PinLayout.Slot(end).Nt;
            
         elseif strcmp(flag,'NcBlock') 

            data.Input.Locator.NcBlock(end).Master=d(i,1);
            data.Input.Locator.NcBlock(end).DomainM=d(i,2);
            data.Input.Locator.NcBlock(end).Pm=d(i,[3 4 5]);
            data.Input.Locator.NcBlock(end).PmReset=data.Input.Locator.NcBlock(end).Pm;

         elseif strcmp(flag,'ClampM') 

            data.Input.Locator.ClampM(end).Master=d(i,1);
            data.Input.Locator.ClampM(end).Slave=d(i,2);
            data.Input.Locator.ClampM(end).DomainM=d(i,3);
            data.Input.Locator.ClampM(end).DomainS=d(i,4);
            data.Input.Locator.ClampM(end).Pm=d(i,[5 6 7]);
            data.Input.Locator.ClampM(end).PmReset=data.Input.Locator.ClampM(end).Pm;
            %--
            nFormat=length(formatFile);
            for k=1:nFormat
               if strcmp(formatFile{k},'Nm')
                  data.Input.Locator.ClampM(end).NormalType{1}=1; % user
                  data.Input.Locator.ClampM(end).Nm=d(i,colPos(k,:));
                  data.Input.Locator.ClampM(end).NmReset=data.Input.Locator.ClampM(end).Nm;
               elseif strcmp(formatFile{k},'Nt')
                  data.Input.Locator.ClampM(end).TangentType{1}=1; % user
                  data.Input.Locator.ClampM(end).Nt=d(i,colPos(k,:));
                  data.Input.Locator.ClampM(end).NtReset=data.Input.Locator.ClampM(end).Nt;
               else 
                   %--
               end
            end

        elseif strcmp(flag,'ClampS') 

            data.Input.Locator.ClampS(end).Master=d(i,1);
            data.Input.Locator.ClampS(end).DomainM=d(i,2);
            data.Input.Locator.ClampS(end).Pm=d(i,[3 4 5]);
            data.Input.Locator.ClampS(end).PmReset=data.Input.Locator.ClampS(end).Pm;
            %--
            nFormat=length(formatFile);
            for k=1:nFormat
               if strcmp(formatFile{k},'Nm')
                  data.Input.Locator.ClampS(end).NormalType{1}=1; % user
                  data.Input.Locator.ClampS(end).Nm=d(i,colPos(k,:));
                  data.Input.Locator.ClampS(end).NmReset=data.Input.Locator.ClampS(end).Nm;
               elseif strcmp(formatFile{k},'Nt')
                  data.Input.Locator.ClampS(end).TangentType{1}=1; % user
                  data.Input.Locator.ClampS(end).Nt=d(i,colPos(k,:));
                  data.Input.Locator.ClampS(end).NtReset=data.Input.Locator.ClampS(end).Nt;
               else 
                   %--
               end
            end
            
        elseif strcmp(flag,'CustomConstraint') 

            data.Input.CustomConstraint(end).Master=d(i,1);
            data.Input.CustomConstraint(end).DomainM=d(i,2);
            data.Input.CustomConstraint(end).Pm=d(i,[3 4 5]);
            data.Input.CustomConstraint(end).PmReset=data.Input.CustomConstraint(end).Pm;
            
          elseif strcmp(flag,'Contact') 

            data.Input.Contact(end).Master=d(i,1);
            data.Input.Contact(end).Slave=d(i,2);

        end
    end
end

%--
function [maxcol, colPos]=format2Columns(maxcol, formatFile)

nFormat=length(formatFile);
colPos=zeros(nFormat,3);
for i=1:nFormat
   if strcmp(formatFile{i},'Nm')
       colPos(i,:)=[maxcol+1 maxcol+2 maxcol+3];
       maxcol=maxcol+3; 
   elseif strcmp(formatFile{i},'Nt')
       colPos(i,:)=[maxcol+1 maxcol+2 maxcol+3];
       maxcol=maxcol+3;
   else

       %---
       % Add here any other option
       %---

   end
end
