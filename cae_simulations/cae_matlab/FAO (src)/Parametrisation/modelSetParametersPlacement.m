% Set placement matrix
function T0w=modelSetParametersPlacement(data,...
                                          parameterValue,...
                                          partID,...
                                          parameterType,...
                                          referenceType,...
                                          T0w)
% Input:
% data: input model
% parameterValue: sampled value
% partID: part ID
% parameterType: parameter type
% referenceType: reference/UCS type
% T0w: Placement matrix

% Output:
% T0w: updated placement matrix
  
%-----------------------------------------
% Update placement   
nsample=length(parameterValue);
for k=1:nsample
    if data.Input.Part(partID).Status==0 && data.Input.Part(partID).Enable % Active part
        %
        % Get placement matrix
        if parameterType==1 % alfa
          Rplc=RodriguesRot([1 0 0],parameterValue(k));
          Pplc=[0;0;0];
        elseif parameterType==2 % beta
          Rplc=RodriguesRot([0 1 0],parameterValue(k));
          Pplc=[0;0;0]; 
        elseif parameterType==3 % gamma
          Rplc=RodriguesRot([0 0 1],parameterValue(k));
          Pplc=[0;0;0];  
        elseif parameterType==4 % deltaX
          Rplc=eye(3,3);
          Pplc=[parameterValue(k);0;0];    
        elseif parameterType==5 % deltaY
          Rplc=eye(3,3);
          Pplc=[0;parameterValue(k);0];    
        elseif parameterType==6 % deltaZ
          Rplc=eye(3,3);
          Pplc=[0;0;parameterValue(k)];    
        end
        %--
        Tplc=eye(4,4);
        Tplc(1:3,1:3)=Rplc; Tplc(1:3,4)=Pplc;

        if referenceType==0 % global UCS
            % Save back
            T0w=Tplc*T0w;
        elseif referenceType==1 % local UCS
            Tucs=data.Input.Part(partID).Placement.UCS;
            % Save back
            T0w=Tucs*Tplc*inv(Tucs)*T0w; %#ok<MINV>
        end
        %
    end
end
%