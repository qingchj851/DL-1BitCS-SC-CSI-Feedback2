function out = SSA_BIHT(Y_real,Y_imag, Address,K,Phi,N,maxiter)

[a,b]=find(Address==-1);
Address(a,b)=0;

H_real = zeros(1,N);
H_imag = zeros(1,N);
ii=0;
htol = 0;
hd_real = Inf;
hd_imag = Inf;

% while(htol < hd_real)&&(ii < maxiter)
while(ii < maxiter)
	% Get gradient
	g_real = (Y_real-sign(H_real*Phi))*Phi';
    g_imag = (Y_imag-sign(H_imag*Phi))*Phi';
	% Step
	a_real = H_real + g_real;
    a_imag = H_imag + g_imag;
	% Best K-term (threshold)
	[~, aidx_real] = sort(abs(a_real), 'descend');  
    a_real(aidx_real(K+1:end)) = 0;
    [~, aidx_imag] = sort(abs(a_imag), 'descend');  
    a_imag(aidx_imag(K+1:end)) = 0;
    % Update x
	H_real = a_real.*Address;
    H_imag = a_imag.*Address;
    ii = ii+1;
%     hd_real = nnz(Y_real - sign(H_real*Phi));
%     hd_imag = nnz(Y_imag - sign(H_imag*Phi));
  
    
end
% out =H_real/(norm(H_real))+(H_imag/(norm(H_imag)))*1i
 H=(H_real+H_imag*1i);
% Now project to sphere
out = H/(norm(H)+1e-10);
% out = Y;
