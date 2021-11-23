clear all;
clc;
close all;


load('train_data_1.mat','CSI_bitR1','US_bitR1','yR1')
load('train_data_2.mat','CSI_bitR2','US_bitR2','yR2')
load('train_data_3.mat','CSI_bitR3','US_bitR3','yR3')
load('train_data_4.mat','CSI_bitR4','US_bitR4','yR4')
load('train_data_5.mat','CSI_bitR5','US_bitR5','yR5')
load('train_data_6.mat','CSI_bitR6','US_bitR6','yR6')
load('train_data_7.mat','CSI_bitR7','US_bitR7','yR7')
load('train_data_8.mat','CSI_bitR8','US_bitR8','yR8')
load('train_data_9.mat','CSI_bitR9','US_bitR9','yR9')
load('train_data_10.mat','CSI_bitR10','US_bitR10','yR10')
load('train_data_11.mat','CSI_bitR11','US_bitR11','yR11')
load('train_data_12.mat','CSI_bitR12','US_bitR12','yR12')
load('train_data_13.mat','CSI_bitR13','US_bitR13','yR13')
load('train_data_14.mat','CSI_bitR14','US_bitR14','yR14')
load('train_data_15.mat','CSI_bitR15','US_bitR15','yR15')
load('train_data_16.mat','CSI_bitR16','US_bitR16','yR16')
load('train_data_17.mat','CSI_bitR17','US_bitR17','yR17')
load('train_data_18.mat','CSI_bitR18','US_bitR18','yR18')
load('train_data_19.mat','CSI_bitR19','US_bitR19','yR19')
load('train_data_20.mat','CSI_bitR20','US_bitR20','yR20')

CSI_bitR(1:3000,:) = CSI_bitR1;
US_bitR(1:3000,:) = US_bitR1;
yR(1:3000,:) = yR1;

CSI_bitR(3001:6000,:) = CSI_bitR2;
US_bitR(3001:6000,:) = US_bitR2;
yR(3001:6000,:) = yR2;

CSI_bitR(6001:9000,:) = CSI_bitR3;
US_bitR(6001:9000,:) = US_bitR3;
yR(6001:9000,:) = yR3;

CSI_bitR(9001:12000,:) = CSI_bitR4;
US_bitR(9001:12000,:) = US_bitR4;
yR(9001:12000,:) = yR4;

CSI_bitR(12001:15000,:) = CSI_bitR5;
US_bitR(12001:15000,:) = US_bitR5;
yR(12001:15000,:) = yR5;

CSI_bitR(15001:18000,:) = CSI_bitR6;
US_bitR(15001:18000,:) = US_bitR6;
yR(15001:18000,:) = yR6;

CSI_bitR(18001:21000,:) = CSI_bitR7;
US_bitR(18001:21000,:) = US_bitR7;
yR(18001:21000,:) = yR7;

CSI_bitR(21001:24000,:) = CSI_bitR8;
US_bitR(21001:24000,:) = US_bitR8;
yR(21001:24000,:) = yR8;

CSI_bitR(24001:27000,:) = CSI_bitR9;
US_bitR(24001:27000,:) = US_bitR9;
yR(24001:27000,:) = yR9;

CSI_bitR(27001:30000,:) = CSI_bitR10;
US_bitR(27001:30000,:) = US_bitR10;
yR(27001:30000,:) = yR10;

CSI_bitR(30001:33000,:) = CSI_bitR11;
US_bitR(30001:33000,:) = US_bitR11;
yR(30001:33000,:) = yR11;

CSI_bitR(33001:36000,:) = CSI_bitR12;
US_bitR(33001:36000,:) = US_bitR12;
yR(33001:36000,:) = yR12;

CSI_bitR(36001:39000,:) = CSI_bitR13;
US_bitR(36001:39000,:) = US_bitR13;
yR(36001:39000,:) = yR13;

CSI_bitR(39001:42000,:) = CSI_bitR14;
US_bitR(39001:42000,:) = US_bitR14;
yR(39001:42000,:) = yR14;

CSI_bitR(42001:45000,:) = CSI_bitR15;
US_bitR(42001:45000,:) = US_bitR15;
yR(42001:45000,:) = yR15;

CSI_bitR(45001:48000,:) = CSI_bitR16;
US_bitR(45001:48000,:) = US_bitR16;
yR(45001:48000,:) = yR16;

CSI_bitR(48001:51000,:) = CSI_bitR17;
US_bitR(48001:51000,:) = US_bitR17;
yR(48001:51000,:) = yR17;

CSI_bitR(51001:54000,:) = CSI_bitR18;
US_bitR(51001:54000,:) = US_bitR18;
yR(51001:54000,:) = yR18;

CSI_bitR(54001:57000,:) = CSI_bitR19;
US_bitR(54001:57000,:) = US_bitR19;
yR(54001:57000,:) = yR19;

CSI_bitR(57001:60000,:) = CSI_bitR20;
US_bitR(57001:60000,:) = US_bitR20;
yR(57001:60000,:) = yR20;

save('train_data.mat','CSI_bitR','US_bitR','yR')

