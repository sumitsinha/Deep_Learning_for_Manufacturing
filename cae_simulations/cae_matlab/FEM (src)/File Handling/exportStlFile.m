% write stl file based on data structure
function exportStlFile(filename,...
                       fem,...
                       ipart)

% open file
idf=fopen(filename,'w');
    
for id=ipart
    
    nameObj=['solid_obj_', num2str(id)];
    fprintf(idf,'solid %s\r\n',nameObj);  

    nele=length(fem.Domain(id).Element);
    for k=1:nele

        ide=fem.Domain(id).Element(k);
        typeele=fem.xMesh.Element(ide).Type;

        if strcmp(typeele,'tria')

                Normal=fem.xMesh.Element(ide).Tmatrix.Normal;
                facek=fem.xMesh.Element(ide).Element;
                vertex=fem.xMesh.Node.Coordinate(facek,:);

                % face 1
                fprintf(idf,'facet normal %.7E %.7E %.7E\r\n', Normal(1),Normal(2),Normal(3) );
                fprintf(idf,'outer loop\r\n');        
                fprintf(idf,'vertex %.7E %.7E %.7E\r\n', vertex(1,:));
                fprintf(idf,'vertex %.7E %.7E %.7E\r\n', vertex(2,:));
                fprintf(idf,'vertex %.7E %.7E %.7E\r\n', vertex(3,:));
                fprintf(idf,'endloop\r\n');
                fprintf(idf,'endfacet\r\n');

        elseif strcmp(typeele,'quad')

                Normal=fem.xMesh.Element(ide).Tmatrix.Normal;
                facek=fem.xMesh.Element(ide).Element;
                vertex=fem.xMesh.Node.Coordinate(facek,:);

                % face 1
                fprintf(idf,'facet normal %.7E %.7E %.7E\r\n', Normal(1),Normal(2),Normal(3) );
                fprintf(idf,'outer loop\r\n');        
                fprintf(idf,'vertex %.7E %.7E %.7E\r\n', vertex(1,:));
                fprintf(idf,'vertex %.7E %.7E %.7E\r\n', vertex(2,:));
                fprintf(idf,'vertex %.7E %.7E %.7E\r\n', vertex(3,:));
                fprintf(idf,'endloop\r\n');
                fprintf(idf,'endfacet\r\n');

                % face 2
                fprintf(idf,'facet normal %.7E %.7E %.7E\r\n', Normal(1),Normal(2),Normal(3) );
                fprintf(idf,'outer loop\r\n');        
                fprintf(idf,'vertex %.7E %.7E %.7E\r\n', vertex(3,:));
                fprintf(idf,'vertex %.7E %.7E %.7E\r\n', vertex(4,:));
                fprintf(idf,'vertex %.7E %.7E %.7E\r\n', vertex(1,:));
                fprintf(idf,'endloop\r\n');
                fprintf(idf,'endfacet\r\n');

        end

    end
    
    fprintf(idf,'endsolid %s\r\n',nameObj);  

end
    


fclose(idf); % close file...


