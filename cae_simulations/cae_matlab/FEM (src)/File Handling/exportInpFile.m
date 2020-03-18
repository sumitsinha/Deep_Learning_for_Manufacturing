% write inp file based on data structure
function exportInpFile(filename,...
                       fem, idpart)

% open file
idf=fopen(filename,'w');

NameObj='** CREATEDinMATLAB';

fprintf(idf,'%s\r\n',NameObj);  

% nodes
for id=idpart
    
    nnode=length(fem.Domain(id).Node);

    fprintf(idf,'*NODE\r\n');  
    for k=1:nnode

        vertex=fem.xMesh.Node.Coordinate(fem.Domain(id).Node(k),:);

        fprintf(idf,'%g , %.7E , %.7E , %.7E\r\n', fem.Domain(id).Node(k), vertex(1),vertex(2),vertex(3));

    end

    % tria
    ntria=length(fem.Domain(id).ElementTria);
    count=1;
    if ntria>0
        fprintf(idf,'*ELEMENT,TYPE=S3\r\n');
        for k=1:ntria
            kk=fem.Domain(id).ElementTria(k);
            elek=fem.xMesh.Element(kk).Element;

            fprintf(idf,'%g , %g , %g , %g\r\n', count, elek(1),elek(2),elek(3));

            count=count+1;
        end
    end

    % quad
    nquad=length(fem.Domain(id).ElementQuad);
    if nquad>0
        fprintf(idf,'*ELEMENT,TYPE=S4\r\n');  
        for k=1:nquad
            kk=fem.Domain(id).ElementQuad(k);
            elek=fem.xMesh.Element(kk).Element;

            fprintf(idf,'%g , %g , %g , %g, %g\r\n', count, elek(1),elek(2),elek(3),elek(4));

            count=count+1;
        end
    end
    
end

% close file
fclose(idf);


