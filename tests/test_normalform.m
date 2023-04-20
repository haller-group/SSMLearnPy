clear all;
load("test_SSMLearn.mat");

etaData = {};
for i=1:2
    etaData{i,1} = times(i,:);
    etaData{i,2} = squeeze(newcoord(i,:,:));
end
ROMOrder = 3;
RDInfo = IMDynamicsFlow(etaData, 'R_PolyOrd', ROMOrder, 'style', 'normalform', 'fig_disp_nfp', 1);