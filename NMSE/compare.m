clear all;
clc;
close all;
warning off;

SNR_dB = 0:2:18;


load('ref[6]_NMSE[10]_160_10.mat')
nmse211 = MSE_final_PRO;
load('ref[6]_NMSE[20]_160_10.mat')
nmse311 = MSE_final_PRO;
load('ref[6]_NMSE[50]_160_10.mat')
nmse411 = MSE_final_PRO;
load('ref[6]_NMSE[100]_160_10.mat')
nmse511 = MSE_final_PRO;
load('proposed_NMSE_160_10.mat')
nmse611 = NMSE;

figure;
a=semilogy(SNR_dB,nmse211,'-.bo',...
     SNR_dB,nmse311,'--MX', SNR_dB,nmse411,'-.Cs',...
     SNR_dB,nmse511,':K',SNR_dB,nmse611,'-ro',...
    'LineWidth',2,'MarkerSize',8);
xlabel('SNR (dB)');
ylabel('NMSE');
axis([0 18 5E-2 1]);

m2=legend('Ref [6] \beta = 10',...
          'Ref [6] \beta = 20','Ref [6] \beta = 50',...
          'Ref [6] \beta = 100','Proposed \beta = 8');

set(m2,'Fontsize',10);
set(gca,'Fontname','Monospaced');%title字体中文显示
grid off;
set(gcf,'position',[200,300,500,430]);

%====================================================
%pho的鲁棒性比较（c2p0.1）
load('ref[6]_NMSE[10]_160_5.mat')
nmse11 = MSE_final_PRO;
load('ref[6]_NMSE[10]_160_10.mat')
nmse21 = MSE_final_PRO;
load('ref[6]_NMSE[10]_160_15.mat')
nmse31 = MSE_final_PRO;


load('proposed_NMSE_160_5.mat')
nmse41 = NMSE;
load('proposed_NMSE_160_10.mat')
nmse51 = NMSE;
load('proposed_NMSE_160_15.mat')
nmse61 = NMSE;


figure;
semilogy(SNR_dB,nmse11,':K+', SNR_dB,nmse21,':rO',  SNR_dB,nmse31,':BS',SNR_dB,nmse41,'-K+',...
     SNR_dB,nmse51,'-rO',...
     SNR_dB,nmse61,'-BS',...
    'LineWidth',2,'MarkerSize',8);
xlabel('SNR (dB)');
ylabel('NMSE');
axis([0 18 5.5E-2 1.05]);

m=legend( 'Ref [6] \rho = 0.05',    'Ref [6] \rho = 0.10',       'Ref [6] \rho = 0.15',      'Proposed \rho = 0.05',...
          'Proposed \rho = 0.10',...
         'Proposed \rho = 0.15');
set(m,'Fontsize',10);
set(gca,'Fontname','Monospaced');%title字体中文显示
grid off;
set(gcf,'position',[200,300,500,430]);


%====================================================
%c的鲁棒性比较
load('ref[6]_NMSE[10]_160_10.mat')
nmse1 = MSE_final_PRO;
load('ref[6]_NMSE[10]_192_10.mat')
nmse2 = MSE_final_PRO;
load('ref[6]_NMSE[10]_224_10.mat')
nmse3 = MSE_final_PRO;

load('proposed_NMSE_160_10.mat')
nmse4 = NMSE;
load('proposed_NMSE_192_10.mat')
nmse5 = NMSE;
load('proposed_NMSE_224_10.mat')
nmse6 = NMSE;



figure;
semilogy(SNR_dB,nmse1,':K*',   SNR_dB,nmse2,':Bs',     SNR_dB,nmse3,':RO',     SNR_dB,nmse4,'-K*',...
      SNR_dB,nmse5,'-Bs',...
     SNR_dB,nmse6,'-RO',...
    'LineWidth',2,'MarkerSize',8);
xlabel('SNR (dB)');
ylabel('NMSE');

axis([0 18 4.3E-2 1]);


m=legend( 'Ref [6] c = 2.0',   'Ref [6] c = 2.5',  'Ref [6] c = 3.0',  'Proposed c = 2.0',...
         'Proposed c = 2.5',...
          'Proposed c = 3.0');
set(m,'Fontsize',10);
set(gca,'Fontname','Monospaced');%title字体中文显示
grid off;
set(gcf,'position',[200,300,500,430]);




