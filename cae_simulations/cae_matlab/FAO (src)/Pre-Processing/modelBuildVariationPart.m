% Build set of variational geometry
function data=modelBuildVariationPart(data)

% No. of instances is defined in "data.Model.Variation.Option.NoSample"
% if "Distribution" of given control point is
    % "Deterministic" => use .Morphing.Parameter(1)
    % "Gaussian" => use random sampling from normal pdf with mean equal to .Morphing.Parameter(1) and std equal to .Morphing.Parameter(2)

%----------
if ~isfield(data.Input,'Part')
    error('Variational geometry (error): No part identified!');
end

%--
npart=length(data.Input.Part);

%--
nosample=data.Model.Variation.Option.NoSample;
%--
for idpart=1:npart
    
     fprintf('Solving morphing mesh: part [%d]\n', idpart);
        
        % Init dataset
        data.Input.Part(idpart).D=[];
        data.Input.Part(idpart).Geometry.Parameter=1;

        % count if any stochastic control point
        if data.Input.Part(idpart).Status==0 && data.Input.Part(idpart).Enable

            r=length(data.Input.Part(idpart).Morphing);
            
            fprintf('      no. control points: %d\n', r);
                         
            % run over all samples
            for k=1:nosample
                
                fprintf('         sample ID: %d\n', k);

                % set deviation at control points
                for kr=1:r
                    distrub=data.Input.Part(idpart).Morphing(kr).Distribution{1};

                    if distrub==1 % deterministic
                        data.Input.Part(idpart).Morphing(kr).DeltaPc=data.Input.Part(idpart).Morphing(kr).Parameter(1);
                    elseif distrub==2 % gaussian
                        mu=data.Input.Part(idpart).Morphing(kr).Parameter(1);
                        sigma=data.Input.Part(idpart).Morphing(kr).Parameter(2);
                        data.Input.Part(idpart).Morphing(kr).DeltaPc=randn(1)*sigma+mu;
                    else
                        
                        %---------------------------------
                        % ADD HERE ANY OTHER pdf
                        %---------------------------------
                        
                    end
                end
            
                % SOLVE DEFORMATION
                [D, flag]=morphGeometrySolve(data, idpart);

                if flag~=1 % failed to build
                    error('Variational geometry (error): Failed to built variation model @ part[%g]', idpart);
                else
                    % save-out
                    data.Input.Part(idpart).D{k}=[D(:,1) D(:,2) D(:,3)]; % [X Y Z]
                end

            end

        end
end

