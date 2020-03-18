% Sample design variables
function X=modelGenerateSamples(data, mlevels, Mlevels, nlevels)

% data: data structure with the following input
    % .Assembly.SamplingStrategy{idS}: option for sampling
        % idS=1 => full factorial: full factorial design
        % idS=2 => random: random sampling
        % idS=3 => user: user table
    % .Assembly.SamplingOptions
        % .SampleSize=sample size (only for opt="random")
        % .IdTable=id of the parameter table (only for opt="user")
% mlevels: list of min values (1xno. of parameters)
% Mlevels: list of max values (1xno. of parameters)
% nlevels: list of levels (1xno. of parameters)

X=[];

samplingOptions=data.Assembly.SamplingOptions;
samplingStrategy=data.Assembly.SamplingStrategy{data.Assembly.SamplingStrategy{1}+1};

if strcmp(samplingStrategy, 'full factorial') 
    
    DoE=fullfact(nlevels);

    % allocate X
    [m, n]=size(DoE);
    X=zeros(m, n);

    for i=1:m     
        for j=1:n
            t=linspace(mlevels(j), Mlevels(j), nlevels(j)); % uniform sample
            X(i,j)=t(DoE(i,j));
        end
    end
    
    
elseif strcmp(samplingStrategy, 'random') 
    
    npara=length(nlevels);
    np=samplingOptions.SampleSize;
    
    X=zeros(np,npara);
    for i=1:npara
        X(:,i)=mlevels(i) + (Mlevels(i)-mlevels(i))*rand(np,1); % random sample
    end
    
elseif strcmp(samplingStrategy, 'user') % use parameter table
    
    % read table
    if samplingOptions.IdTable==0
        error('Generating parameter (error): invalid Table ID');
    end
    
    % check consistency of dataset
    if ~isfield(data.Input, 'Parameter')
        error('Generating parameter (error): no "Parameter" table identified');
    end
    if samplingOptions.IdTable>length(data.Input.Parameter)
        error('Generating parameter (error): inconsistency between Table ID and Parameter Table');
    end
    
    % get "X"
    X=data.Input.Parameter(samplingOptions.IdTable).X;
    
    % check consistency of dataset in "X"
    if size(X,2)~=length(nlevels) || isempty(X)
        error('Generating parameter (error): inconsistent dataset in Parameter Table');
    end
    
end

%--------------
    %--------------
    % add any other sampling strategy here
    %--------------
%--------------

