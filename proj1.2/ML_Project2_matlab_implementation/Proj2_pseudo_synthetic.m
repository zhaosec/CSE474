load('synthetic.mat');
NUM2=x';
NUM=[t,NUM2];

width=size(NUM,2);
Min_Error=10000;
Min_M=3;
Min_lamb=0.02;

for M=15:1:20
    for lamb=0.02:0.02:0.1

leng=size(NUM,1);
train_leng=round(9/10*leng);
vali_leng=round(1/10*leng);
%test_leng=leng-train_leng-vali_leng;
max_iters=100;
Data_Train=NUM(1:train_leng,2:width);
Label_Train=NUM(1:train_leng,1);
Data_Vali=NUM(train_leng:(train_leng+vali_leng),2:width);
Label_Vali=NUM(train_leng:(train_leng+vali_leng),1);
%Data_Test=NUM((train_leng+vali_leng):leng,2:47);
%Label_Test=NUM((train_leng+vali_leng):leng,1);
D=size(Data_Train,2);

Init_Centers=kMeansInitCentroids(Data_Train, M-1);
[Centers, Memberships]=kMeans(Data_Train, Init_Centers, max_iters);

Sigm_Gaus_Temp=Sigm_Gaussian(Data_Train, Centers, Memberships);
for q=1:1:M
    for p=1:1:D
        if(abs(Sigm_Gaus_Temp(p,p,q))<0.05)
            Sigm_Gaus_Temp(p,p,q)=0.2;
        end
    end
end
Sigm_Gaus=zeros(D,D,M);
for i=1:1:M
    for j=1:1:D
        


    end
end

Phi=zeros(size(Data_Train,1),M);
%One=ones(size(Data_Train,1),1);
%Phi=[One,Phi];
 for i=1:1:size(Phi,1)
    for j=1:1:(size(Phi,2))
    
       Phi(i,j)=Gaus_Dis(Data_Train(i,:),Centers(j,:),Sigm_Gaus(:,:,j));
       
   end
 end

 HH=ones(1,M);
 I=diag(HH);
 W=inv((I.*lamb+Phi'*Phi))*Phi'*Label_Train;
 
 error=0;
 for i=1:1:size(Data_Vali,1)
     
     Data_Temp=[1,Data_Vali(i,:)];

     Y=Cal_Y(Data_Vali(i),W,Centers,Sigm_Gaus);
     
     error=error+0.5*(Y-Label_Vali(i))^2;
     
     
 end

      ER=sqrt(2*error/(size(Data_Vali,1)));
      if(Min_Error>ER)
          Min_Error=ER;
          Min_M=M;
          Min_lamb=lamb;
      end
 
 
 
 
 
    end
end
 



[NUM1,TXT1,RAW1]=xlsread('Data.xls');
M=Min_M;
lamb=Min_lamb;

leng=size(NUM,1);
leng1=size(NUM1,1);
train_leng=round(9/10*leng);
train_leng1=round(9/10*leng1);
vali_leng=round(1/10*leng);
vali_leng1=round(1/10*leng1);
test_leng=leng-train_leng-vali_leng;
max_iters=100;
Data_Train=NUM(1:train_leng,2:width);
Label_Train=NUM(1:train_leng,1);
Data_Vali=NUM(train_leng:(train_leng+vali_leng),2:width);
Label_Vali=NUM(train_leng:(train_leng+vali_leng),1);
Data_Test=NUM((train_leng+vali_leng):leng,2:width);
Label_Test=NUM((train_leng+vali_leng):leng,1);
D=size(Data_Train,2);

Init_Centers=kMeansInitCentroids(Data_Train, M);
[Centers, Memberships]=kMeans(Data_Train, Init_Centers, max_iters);

Sigm_Gaus=Sigm_Gaussian(Data_Train, Centers, Memberships);
for q=1:1:M
    for p=1:1:D
        if(abs(Sigm_Gaus(p,p,q))<0.05)
            Sigm_Gaus(p,p,q)=0.2;
        end
    end
end


Phi=zeros(size(Data_Train,1),M);
%One=ones(size(Data_Train,1),1);
%Phi=[One,Phi];
 for i=1:1:size(Phi,1)
    for j=1:1:(size(Phi,2)-1)
    
       Phi(i,j+1)=Gaus_Dis(Data_Train(i,:),Centers(j,:),Sigm_Gaus(:,:,j));
       
   end
 end
 HH=ones(1,M);
 I=diag(HH);
 W=inv((I.*lamb+Phi'*Phi))*Phi'*Label_Train;
 
 error=0;
 for i=1:1:size(Data_Vali,1)
     
     Data_Temp=[1,Data_Vali(i,:)];

     Y=Cal_Y(Data_Vali(i),W,Centers,Sigm_Gaus);
     
     error=error+0.5*(Y-Label_Vali(i))^2;
     
     
 end

      ER=sqrt(2*error/(size(Data_Vali,1)));
    
 
 error=0;
 for i=1:1:size(Data_Train,1)
     
     Data_Temp=[1,Data_Train(i,:)];

     Y=Cal_Y(Data_Train(i),W,Centers,Sigm_Gaus);
     
     error=error+0.5*(Y-Label_Train(i))^2;
     
     
 end

      ER_Train=sqrt(2*error/(size(Data_Vali,1)));
      
      
      
      
      
 
 w2=W;
 M2=M;
 mu2=Centers';
 Sigma2=Sigm_Gaus;
 lambda2=lamb;
 trainInd2=linspace(1,train_leng,train_leng)';
 validInd2=linspace((train_leng+1),(train_leng+vali_leng),vali_leng)';
 validPer2=ER;
 trainPer2=ER_Train;
 
 


