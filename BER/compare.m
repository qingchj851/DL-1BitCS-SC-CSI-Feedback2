clear all;
clc;
close all;
warning off;

SNR_dB = 0:2:18;


load('ref[6]_BER_160_5.mat')
yulber1 = ulber;
ydlber1 = dlber;
load('ref[6]_BER_160_10.mat')
yulber2 = ulber;
ydlber2 = dlber;
load('ref[6]_BER_160_15.mat')
yulber3 = ulber;
ydlber3 = dlber;
load('ref[6]_BER_192_10.mat')
yulber4 = ulber;
ydlber4 = dlber;
load('ref[6]_BER_224_10.mat')
yulber5 = ulber;
ydlber5 = dlber;


load('proposed_BER_CSI_dim160_5.mat')
culber1 = US_BER;
cdlber1 = CSI_BER;
load('proposed_BER_CSI_dim160_10.mat')
culber2 = US_BER;
cdlber2 = CSI_BER;
load('proposed_BER_CSI_dim160_15.mat')
culber3 = US_BER;
cdlber3 = CSI_BER;
load('proposed_BER_CSI_dim192_10.mat')
culber4 = US_BER;
cdlber4 = CSI_BER;
load('proposed_BER_CSI_dim224_10.mat')
culber5 = US_BER;
cdlber5 = CSI_BER;

%总性能比较（c=2,p=0.1）
figure;
semilogy(SNR_dB,yulber2,'--bs',SNR_dB,culber2,'--rs',...
     SNR_dB,ydlber2,'-bo', SNR_dB,cdlber2,'-ro',...
    'LineWidth',2,'MarkerSize',8);
xlabel('SNR (dB)');
ylabel('BER');

m=legend( 'Ref[6] UL\_US','Proposed UL\_US',...
          'Ref[6] MFV','Proposed MFV');
set(m,'Fontsize',10);
set(gca,'Fontname','Monospaced');%title字体中文显示
grid off;
axis([0 18 4.5e-8 0.4])
set(gcf,'position',[200,300,500,430]);

%p的鲁棒性性能比较（c=2,p=0.05，0.1，0.15）
figure;
semilogy(SNR_dB,yulber1,'--ro',SNR_dB,culber1,'-ro',...
     SNR_dB,yulber2,'--ks',SNR_dB,culber2,'-ks',...
     SNR_dB,yulber3,'--b^',SNR_dB,culber3,'-b^',...
     SNR_dB,ydlber1,'--rv',SNR_dB,cdlber1,'-rv',...
     SNR_dB,ydlber2,'--k>',SNR_dB,cdlber2,'-k>',...
     SNR_dB,ydlber3,'--b<',SNR_dB,cdlber3,'-b<',...
     'LineWidth',2,'MarkerSize',8);
xlabel('SNR (dB)');
ylabel('BER');

m=legend( 'Ref[6] UL\_US \rho=0.05','Proposed UL\_US \rho=0.05',...
          'Ref[6] UL\_US \rho=0.10','Proposed UL\_US \rho=0.10',...
          'Ref[6] UL\_US \rho=0.15','Proposed UL\_US \rho=0.15',...
          'Ref[6] MFV \rho=0.05','Proposed MFV \rho=0.05',...
          'Ref[6] MFV \rho=0.10','Proposed MFV \rho=0.10',...
          'Ref[6] MFV \rho=0.15','Proposed MFV \rho=0.15');
set(m,'Fontsize',10);
set(gca,'Fontname','Monospaced');%title字体中文显示
grid off;
axis([0 18 9e-9 0.4])
set(gcf,'position',[200,300,500,430]);


%c的鲁棒性性能比较（c=2，2.5，3，p=0.1）
figure;
semilogy(SNR_dB,yulber2,'--ro',SNR_dB,culber2,'-ro',...
     SNR_dB,yulber4,'--ks',SNR_dB,culber4,'-ks',...
     SNR_dB,yulber5,'--b^',SNR_dB,culber5,'-b^',...
     SNR_dB,ydlber2,'--rv',SNR_dB,cdlber2,'-rv',...
     SNR_dB,ydlber4,'--k<',SNR_dB,cdlber4,'-k<',...
     SNR_dB,ydlber5,'--b>',SNR_dB,cdlber5,'-b>',...
     'LineWidth',2,'MarkerSize',8);
xlabel('SNR (dB)');
ylabel('BER');

m=legend( 'Ref[6] UL\_US c=2','Proposed UL\_US c=2',...
          'Ref[6] UL\_US c=2.5','Proposed UL\_US c=2.5',...
          'Ref[6] UL\_US c=3','Proposed UL\_US c=3',...
          'Ref[6] MFV c=2','Proposed MFV c=2',...
          'Ref[6] MFV c=2.5','Proposed MFV c=2.5',...
          'Ref[6] MFV c=3','Proposed MFV c=3');
set(m,'Fontsize',10);
set(gca,'Fontname','Monospaced');%title字体中文显示
grid off;
axis([0 18 4.5e-8 0.4])
set(gcf,'position',[200,300,500,430]);

