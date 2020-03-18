function data=runSimulationCoreVariationGeometry(data, idparts, opt)

% INPUT
% data: data structure
% idparts: list of parts
% opt.
    %. Flag: "deterministic"/"stochastic" solution
    %. Parameter: parameter related to pol chaos expansion

% OUTPUT
% data: data structure

% loop over all parts
fprintf('Generating variational geometry...\n');  

if strcmp(opt.Flag,'deterministic') % deterministic
    data=localDeterministic(data, idparts);
else
    data=localStochastic(data, idparts, opt.Parameter);
end


% local functions
%----------------------------
function data=localStochastic(data, idparts, para)

nSip=data.Assembly.PolChaos.nSip;
Csi=data.Assembly.PolChaos.Csi;

c=1;
for idpart=idparts

     fprintf('      Solving part: %g\n', idpart);  
        
     if nSip(c)>0 % stochastic part

         fprintf('        Type of deformation: "Morphed"\n');  
         
         r=length(data.Input.Part(idpart).Morphing);

         % get deviation at control points
         cr=1;
         for k=1:r
            distrub=data.Input.Part(idpart).Morphing(k).Distribution{1};

            if distrub==1 % deterministic
                data.Input.Part(idpart).Morphing(k).DeltaPc=data.Input.Part(idpart).Morphing(k).Parameter(1);
            elseif distrub==2 % gaussian
                data.Input.Part(idpart).Morphing(k).DeltaPc=Csi{para,c}(cr);
                
                cr=cr+1;
            else


                % ADD HERE ANY OTHER pdf

            end

         end
         
         % solve...
        [D, flag]=morphGeometrySolve(data, idpart);
        
        if flag==1 % solved
            data.Input.Part(idpart).D{para}=[D(:,1) D(:,2) D(:,3)]; % [X Y Z]
            data.Input.Part(idpart).Geometry.Parameter=para;
        else
            data.Input.Part(idpart).Geometry.Type{1}=1; % use nominal
            Error('Failed to build variational geometry @ part[%g]', idpart);  
        end
        
     elseif nSip(c)==0 % deterministic part
                  
         data=localDeterministic(data, idpart);
     end
     
     c=c+1;
end
    

%----------------------------
function data=localDeterministic(data, idparts)

for idpart=idparts
    
    fprintf('      Solving part: %g\n', idpart);  
    
    gpart=data.Input.Part(idpart).Geometry.Type{1}; % part geometry
    
    if gpart==2  % morphed
        
        fprintf('        Type of geometry: "Morphed"\n');  
        
        r=length(data.Input.Part(idpart).Morphing);
        for k=1:r
            data.Input.Part(idpart).Morphing(k).DeltaPc=data.Input.Part(idpart).Morphing(k).Parameter(1);
        end
        
        % solve...
        [D, flag]=morphGeometrySolve(data, idpart);
        
        if flag==1 % solved
            data.Input.Part(idpart).D{1}=[D(:,1) D(:,2) D(:,3)]; % [X Y Z]
            data.Input.Part(idpart).Geometry.Parameter=1;
        else
            error('Failed to build variational geometry @ part[%g]', idpart);  
        end
        
    elseif gpart==3  % measured
        
        fprintf('        Type of geometry: "Measured"\n');  
        
        ppart=data.Input.Part(idpart).Geometry.Parameter;
        
        if ppart==0 || ppart>length(data.Input.Part(idpart).D)
           error('Failed to build variational geometry @ part[%g]', idpart);  
        end
        
    end
    
end

