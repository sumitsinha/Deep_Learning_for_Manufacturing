% write bdf file based on data structure
function exportBdfFile(filename,...
                      fem,...
                      idpart)

% use the standard 8-digit format
% open file
idf=fopen(filename,'w');

fprintf(idf,'$$ Created in MATLAB\r\n');  
fprintf(idf,'$$---------------------$$\r\n');  
fprintf(idf,'BEGIN BULK \r\n');  
       

for id=idpart
    
    % nodes
    nnode=length(fem.Domain(id).Node);
    
    for k=1:nnode

        vertex=fem.xMesh.Node.Coordinate(fem.Domain(id).Node(k),:);

        v1=num2str(vertex(1),'%.16f');
        v1=v1(1:8);

        v2=num2str(vertex(2),'%.16f');
        v2=v2(1:8);

        v3=num2str(vertex(3),'%.16f');
        v3=v3(1:8);
        fprintf(idf,'GRID    %8i%8s%s%s%s\r\n',k,'',v1,v2,v3);

    end

    % element
    % tria
    ntria=length(fem.Domain(id).ElementTria);
    count=1;
    for k=1:ntria
        kk=fem.Domain(id).ElementTria(k);
        elek=fem.xMesh.Element(kk).Element;

        fprintf(idf,'CTRIA3  %8i%8i%8i%8i%8i\r\n',count,0,elek(1),elek(2),elek(3));

        count=count+1;
    end

    % quad
    nquad=length(fem.Domain(id).ElementQuad);
    for k=1:nquad
        kk=fem.Domain(id).ElementQuad(k);
        elek=fem.xMesh.Element(kk).Element;

        fprintf(idf,'CQUAD4  %8i%8i%8i%8i%8i%8i\r\n',count,0,elek(1),elek(2),elek(3),elek(4));

        count=count+1;
    end

end
    
fprintf(idf,'ENDDATA \r\n');  

fclose(idf); % close file





