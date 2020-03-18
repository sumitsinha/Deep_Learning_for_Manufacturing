% write data for mesh denoising
function fem=femmesh2triamesh(fem, idparts)

disp('Saving data for mesh denoising...')

%--
if fem.Options.ConnectivityUpdate
    
    % save faces
    
    disp('>>---')
    disp('Generating triangular faces...')

        ntria=0;
        for ip=idparts
            ntria = ntria+ length(fem.Domain(ip).ElementTria) + 2*length(fem.Domain(ip).ElementQuad);
        end
        fem.Denoise.Tria=zeros(ntria,3);

        count=1;
        for ip=idparts

            % set intitial
            fem.Denoise.Domain(ip).Tria=[];

            nele=length(fem.Domain(ip).Element);
            for k=1:nele

                ide=fem.Domain(ip).Element(k);
                typeele=fem.xMesh.Element(ide).Type;

                if strcmp(typeele,'tria')

                        fem.Denoise.Domain(ip).Tria=[fem.Denoise.Domain(ip).Tria, count];

                        %--
                        fem.Denoise.Tria(count,:)=fem.xMesh.Element(ide).Element;
                        count=count+1;

                elseif strcmp(typeele,'quad')

                        facek=fem.xMesh.Element(ide).Element;

                        fem.Denoise.Domain(ip).Tria=[fem.Denoise.Domain(ip).Tria, count];
                        %--
                        fem.Denoise.Tria(count,:)=facek([1 2 3]);
                        count=count+1;

                        fem.Denoise.Domain(ip).Tria=[fem.Denoise.Domain(ip).Tria, count];
                        %--
                        fem.Denoise.Tria(count,:)=facek([3 4 1]);
                        count=count+1;
                end

            end

        end

        % get connectivity
        nnode=size(fem.xMesh.Node.Coordinate,1);

        disp('Generating node-to-element connectivity...')
        fem.Denoise.Connectivity.Node2Ele=node2element(fem.Denoise.Tria, nnode);

        disp('Generating element-to-element connectivity...')
        fem.Denoise.Connectivity.Ele2Ele=element2element(fem.Denoise.Tria,...
                                                         fem.Denoise.Connectivity.Node2Ele);
                                             
    
        % get normal vectors
        fem.Denoise.Trianormal=trinormal(fem.Denoise.Tria,...
                                         fem.xMesh.Node.Coordinate);

end


% this function get the connected element for every node
function n2e=node2element(face, nnode)

n2e=cell(1,nnode);

nele=size(face,1);

for k=1:nele
    facek=face(k,:);
    
    for i=1:3
        n2e{facek(i)}=[n2e{facek(i)},k];
    end
    
end

% this function get the connected elements for every element
function e2e=element2element(face, n2e)

nele=size(face,1);

e2e=cell(1,nele);

for k=1:nele
    
    facek=face(k,:);
    
    count=0;
    for j=1:3
        count=count + length(n2e{facek(j)});
    end
    
    temp=zeros(1,count);
    ce=0;
    for j=1:3
        cs=ce+1;       
        ce=ce+length(n2e{facek(j)});
                
        temp(cs:ce)=n2e{facek(j)};
    end
        
    e2e{k}=unique(temp);
    
end


    


