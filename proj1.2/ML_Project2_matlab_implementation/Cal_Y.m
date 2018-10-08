 function Y = Cal_Y(Data_Test,W, Centers, Sigm)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
  M=size(Centers,1);
  D=size(Centers,2);
  Dis=zeros(1,M);

  for i=1:1:M
      Dis(1,i)=Gaus_Dis(Data_Test,Centers(i,:),Sigm(:,:,i));      
      
  end
  
    Dis(1,1)=1;
 
  Y=W'*Dis';
  %disp(Y);
 
  
end