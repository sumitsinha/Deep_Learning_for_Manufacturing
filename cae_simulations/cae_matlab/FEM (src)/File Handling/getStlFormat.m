% identify slt format
function stltype = getStlFormat(filename)

% Open the file:
fidIN = fopen(filename);

% Check the file size first, since binary files MUST have a size of 84+(50*n)
fseek(fidIN,0,1);         % Go to the end of the file
fidSIZE = ftell(fidIN);   % Check the size of the file

if rem(fidSIZE-84,50) > 0
    
  stltype = 'ascii';

else

  % Files with a size of 84+(50*n), might be either ascii or binary...
    
  % Read first 80 characters of the file.
  % For an ASCII file, the data should begin immediately (give or take a few
  % blank lines or spaces) and the first word must be 'solid'.
  % For a binary file, the first 80 characters contains the header.
  % It is bad practice to begin the header of a binary file with the word
  % 'solid', so it can be used to identify whether the file is ASCII or
  % binary.
  fseek(fidIN,0,-1);        % Go to the start of the file
  firsteighty = char(fread(fidIN,80,'uchar')');

  % Trim leading and trailing spaces:
  firsteighty = strtrim(firsteighty);

  % Take the first five remaining characters, and check if these are 'solid':
  firstfive = firsteighty(1:min(5,length(firsteighty)));

  % Double check by reading the last 80 characters of the file.
  % For an ASCII file, the data should end (give or take a few
  % blank lines or spaces) with 'endsolid <object_name>'.
  % If the last 80 characters contains the word 'endsolid' then this
  % confirms that the file is indeed ASCII.
  if strcmp(firstfive,'solid')
  
    fseek(fidIN,-80,1);     % Go to the end of the file minus 80 characters
    lasteighty = char(fread(fidIN,80,'uchar')');
  
    if findstr(lasteighty,'endsolid')
      stltype = 'ascii';
    else
      stltype = 'binary';
    end
  
  else
    stltype = 'binary';
  end
  
end

%--
fclose(fidIN);
