function [M,lamb]=Select_M_lamb()


load('synthetic.mat');
NUM2=x';
NUM=[t,NUM2];

width=size(NUM,2);


max_iters=200;
leng=size(NUM,1);
train_leng=round(9/10*leng);
vali_leng=round(1/10*leng);
Data_Train=NUM(1:train_leng,2:width);
Label_Train=NUM(1:train_leng,1);
Data_Vali=NUM((train_leng+1):(train_leng+vali_leng),2:width);
Label_Vali=NUM((train_leng+1):(train_leng+vali_leng),1);
D=size(Data_Train,2);

Min_M=7;
Min_lamb=0.01;
Min_error=10000;



for M=10:1:16;
    for lamb=0.01:0.03:0.13

Init_Centers=kMeansInitCentroids(Data_Train, M);
[Centers, Memberships]=kMeans(Data_Train, Init_Centers, max_iters);

Sigm_Gaus=Sigm_Gaussian(Data_Train, Centers, Memberships);
%{
Sigm_Gaus=zeros(D,D,M);
var=var_column(Data_Train);
for i=1:1:M
   for j=1:1:D
       Sigm_Gaus(j,j,i)=0.1*var(1,j);
   
   end
end
%}

Phi=zeros(size(Data_Train,1),M);
 for i=1:1:size(Phi,1)
    for j=1:1:(size(Phi,2))
      
        Phi(i,j)=Gaus_Dis(Data_Train(i,:),Centers(j,:),Sigm_Gaus(:,:,j));
        %disp(Sigm_Gaus(:,:,j));
        
        
   end
 end

 for i=1:1:size(Phi,1)
       Phi(i,1)=1;
 end
 HH=ones(1,M);
 I=diag(HH);
 
 W=inv((I.*lamb+Phi'*Phi))*Phi'*Label_Train;
 

  
  error=0;
 for i=1:1:size(Data_Vali,1)
     

     Y=Cal_Y(Data_Vali(i),W,Centers,Sigm_Gaus);
     
     error=error+0.5*((Y-Label_Vali(i))^2);
     
     
     
 end

 if(Min_error>error)
     Min_error=error;
     Min_M=M;
     Min_lamb=lamb;
 end
 
    end
 
end
     E=error;%+(W')*W*0.5*lamb;
 
     ER=sqrt(2*E/(size(Data_Vali,1)));
     validPer2=ER;
     M=Min_M;
     lamb=Min_lamb;

















end
