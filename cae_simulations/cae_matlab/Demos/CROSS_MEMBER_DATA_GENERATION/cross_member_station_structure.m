function stationData=cross_member_station_structure ()

%% Please Verify If station Structure is correct

%% STATION 1

% Stage[1]: non-ideal
stationID=1;
stationData(stationID)=initStation();
stationData(stationID).Part=[1 2 3 4];   
stationData(stationID).Type{1}= 0; % non-ideal

% Stage[2]: Pinhole Positioning Part 1 and Part 2
stationID=2;
stationData(stationID)=initStation();
stationData(stationID).Part=[1 2];  
stationData(stationID).PinHole=[1 2 3 4];
stationData(stationID).Type{1}=2; % pin placement

% Stage[3]: Clamping
stationID=3;
stationData(stationID)=initStation();
stationData(stationID).Part=[1 2];   
stationData(stationID).PinHole=[1 2 3 4];
stationData(stationID).ClampS=[1 2 3 4 5];
stationData(stationID).CustomConstraint=[];
stationData(stationID).Contact=[1]; 
stationData(stationID).Type{1}=3; % clamp

% Stage[4]: Fasten
stationID=4;
stationData(stationID)=initStation();
stationData(stationID).Part=[1 2];   
stationData(stationID).PinHole=[1 2 3 4];
stationData(stationID).ClampS=[1 2 3 4 5];
stationData(stationID).ClampM=[3:10];
stationData(stationID).CustomConstraint=[];
stationData(stationID).Contact=[1]; 
stationData(stationID).Type{1}=3; % fasten

% Stage[5]: Release
stationID=5;
stationData(stationID)=initStation();
stationData(stationID).Part=[1 2];   
stationData(stationID).PinHole=[1 3];
stationData(stationID).ClampS=[3 4];
stationData(stationID).CustomConstraint=[];
stationData(stationID).Stitch=[1:8];
stationData(stationID).Contact=[1]; 
stationData(stationID).Type{1}=4; % release

%%
% STATION 2

% Stage[6]: Pinhole Positioning Part 3 and Part 4
stationID=6;
stationData(stationID)=initStation();
stationData(stationID).Part=[3 4]; 
stationData(stationID).PinHole=[5 6];
stationData(stationID).PinSlot=[1 2];
stationData(stationID).UCS.Source=3;
stationData(stationID).UCS.Destination=4;
stationData(stationID).Type{1}=2; % pin placement

% Stage[7]:  clamp
stationID=7;
stationData(stationID)=initStation();
stationData(stationID).Part=[3 4];   
stationData(stationID).PinHole=[5 6];
stationData(stationID).PinSlot=[1 2];
stationData(stationID).NcBlock=[1];
stationData(stationID).ClampS=[6 7 8 9];
stationData(stationID).ClampM=[1 2];
stationData(stationID).Contact=[2]; 
stationData(stationID).Type{1}=3; % clamp

% Stage[8]: fasten
stationID=8;
stationData(stationID)=initStation();
stationData(stationID).Part=[3 4];   
stationData(stationID).PinHole=[5 6];
stationData(stationID).PinSlot=[1 2];
stationData(stationID).NcBlock=[1];
stationData(stationID).ClampS=[6 7 8 9];
stationData(stationID).ClampM=[1 2 11:20];
stationData(stationID).Contact=[2]; 
stationData(stationID).Type{1}=3; % fasten

% Stage[9]: release
stationID=9;
stationData(stationID)=initStation();
stationData(stationID).Part=[3 4];   
stationData(stationID).PinHole=[5];
stationData(stationID).PinSlot=[1];
stationData(stationID).NcBlock=[];
stationData(stationID).ClampS=[6 7 8 9];
stationData(stationID).Stitch=[9:18];
stationData(stationID).Contact=[2]; 
stationData(stationID).Type{1}=4; % release
%

%% STATION 3
% Stage[10]: Place
stationID=10;
stationData(stationID)=initStation();
stationData(stationID).Part=[1 2 3 4];   
stationData(stationID).PinHole=[1 3 5 6];
stationData(stationID).PinSlot=[1];
stationData(stationID).NcBlock=[];
stationData(stationID).CustomConstraint=[];
stationData(stationID).Stitch=[1:18];
stationData(stationID).Contact=[3]; 
stationData(stationID).UCS.Source=[1 3];
stationData(stationID).UCS.Destination=[2 4];
stationData(stationID).Type{1}=2; % place on new locators

% Stage[11]: Clamp
stationID=11;
stationData(stationID)=initStation();
stationData(stationID).Part=[1 2 3 4];   
stationData(stationID).PinHole=[1 3 5];
stationData(stationID).PinSlot=[1];
stationData(stationID).ClampS=[3 4 6 7 8 9];
stationData(stationID).CustomConstraint=[];
stationData(stationID).Stitch=[1:18];
stationData(stationID).Contact=[3]; 
stationData(stationID).Type{1}=3; % clamp

% Stage[12]: Fasten
stationID=12;
stationData(stationID)=initStation();
stationData(stationID).Part=[1 2 3 4];   
stationData(stationID).PinHole=[1 3 5];
stationData(stationID).PinSlot=[1];
stationData(stationID).ClampS=[3 4 6 7 8 9];
stationData(stationID).ClampM=[21:27];
stationData(stationID).CustomConstraint=[];
stationData(stationID).Stitch=[1:18];
stationData(stationID).Contact=[3]; 
stationData(stationID).Type{1}=3; % fasten

% Stage[13]: Release
stationID=13;
stationData(stationID)=initStation();
stationData(stationID).Part=[1 2 3 4];   
stationData(stationID).PinHole=[5];
stationData(stationID).PinSlot=[1];
stationData(stationID).ClampS=[8 9];
stationData(stationID).CustomConstraint=[];
stationData(stationID).Stitch=[1:25];
stationData(stationID).Contact=[3]; 
stationData(stationID).Type{1}=4; % release
%

end