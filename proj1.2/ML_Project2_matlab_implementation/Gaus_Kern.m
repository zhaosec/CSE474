function Gaus_kern = Gaus_Kern(Data, Centers, Sigm)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
M=size(Centers,1);
Gaus_kern=zeros(1,M);
for i=1:1:M
    Gaus_kern(1,i)=Gaus_Dis(Data,Centers(i,:),Sigm(:,:,i))*Gaus_Cons(Sigm(:,:,1));
    
end
    %disp(Gaus_Cons(Sigm(:,:,1)));
    Gaus_kern(1,1)=1;
    
end


