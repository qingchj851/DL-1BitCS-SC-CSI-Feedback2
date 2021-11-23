clear all;
clc;
close all;

load('testdata_10_1.mat','CSI_bitR1','US_bitR1','yR1')
load('testdata_10_2.mat','CSI_bitR2','US_bitR2','yR2')
load('testdata_10_3.mat','CSI_bitR3','US_bitR3','yR3')
load('testdata_10_4.mat','CSI_bitR4','US_bitR4','yR4')
load('testdata_10_5.mat','CSI_bitR5','US_bitR5','yR5')
load('testdata_10_6.mat','CSI_bitR6','US_bitR6','yR6')
load('testdata_10_7.mat','CSI_bitR7','US_bitR7','yR7')
load('testdata_10_8.mat','CSI_bitR8','US_bitR8','yR8')

CSI_bitR(1:2500,:) = CSI_bitR1;
US_bitR(1:2500,:) = US_bitR1;
yR(1:2500,:) = yR1;

CSI_bitR(2501:5000,:) = CSI_bitR2;
US_bitR(2501:5000,:) = US_bitR2;
yR(2501:5000,:) = yR2;

CSI_bitR(5001:7500,:) = CSI_bitR3;
US_bitR(5001:7500,:) = US_bitR3;
yR(5001:7500,:) = yR3;

CSI_bitR(7501:10000,:) = CSI_bitR4;
US_bitR(7501:10000,:) = US_bitR4;
yR(7501:10000,:) = yR4;

CSI_bitR(10001:12500,:) = CSI_bitR5;
US_bitR(10001:12500,:) = US_bitR5;
yR(10001:12500,:) = yR5;

CSI_bitR(12501:15000,:) = CSI_bitR6;
US_bitR(12501:15000,:) = US_bitR6;
yR(12501:15000,:) = yR6;

CSI_bitR(15001:17500,:) = CSI_bitR7;
US_bitR(15001:17500,:) = US_bitR7;
yR(15001:17500,:) = yR7;

CSI_bitR(17501:20000,:) = CSI_bitR8;
US_bitR(17501:20000,:) = US_bitR8;
yR(17501:20000,:) = yR8;

save('testdata_10.mat','CSI_bitR','US_bitR','yR')