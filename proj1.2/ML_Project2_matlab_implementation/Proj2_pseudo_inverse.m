[NUM,TXT,RAW]=xlsread('Data.xls');


Min_Error=10000;
Min_M=3;
Min_lamb=0.02;

%for M=12:1:18
 %   for lamb=0.08:0.02:0.2

 M=14;
    lamb=0.16;
 
leng=size(NUM,1);
train_leng=round(8/10*leng);
vali_leng=round(1/10*leng);
test_leng=leng-train_leng-vali_leng;
max_iters=100;
Data_Train=NUM(1:train_leng,2:47);
Label_Train=NUM(1:train_leng,1);
Data_Vali=NUM(train_leng:(train_leng+vali_leng),2:47);
Label_Vali=NUM(train_leng:(train_leng+vali_leng),1);
Data_Test=NUM((train_leng+vali_leng):leng,2:47);
Label_Test=NUM((train_leng+vali_leng):leng,1);
D=size(Data_Train,2);

Init_Centers=kMeansInitCentroids(Data_Train, M-1);
[Centers, Memberships]=kMeans(Data_Train, Init_Centers, max_iters);

Sigm_Gaus=Sigm_Gaussian(Data_Train, Centers, Memberships);
for q=1:1:M-1
    for p=1:1:D
        if(abs(Sigm_Gaus(p,p,q))<0.05)
            Sigm_Gaus(p,p,q)=0.2;
        end
    end
end


Phi=zeros(size(Data_Train,1),M-1);
One=ones(size(Data_Train,1),1);
Phi=[One,Phi];
 for i=1:1:size(Phi,1)
    for j=1:1:(size(Phi,2)-1)
    
       Phi(i,j+1)=Gaus_Dis(Data_Train(i,:),Centers(j,:),Sigm_Gaus(:,:,j));
       
    end
 end

 I=diag((M));
 W=inv((I.*lamb+Phi'*Phi))*Phi'*Label_Train;
 
 error=0;
 for i=1:1:size(Data_Vali,1)
     
     Data_Temp=[1,Data_Vali(i,:)];

     Y=Cal_Y(Data_Vali(i),W,Centers,Sigm_Gaus);
     
     error=error+0.5*(Y-Label_Vali(i))^2;
     
     
 end
 
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
 
 
 
 
 
 
w1=W;
M1=M;
 mu1=[ones(1,46);Centers]';
 Sigma1=zeros(D,D,M);
 for i=1:1:D
     for j=1:1:M
         if(j==1)
         Sigma1(i,i,j)=1;
         else
             Sigma1(i,i,j)=Sigm_Gaus(i,i,j-1);
         end
     end
 end
 lambda1=lamb;
 trainInd1=linspace(1,train_leng,train_leng)';
 validInd1=linspace((train_leng+1),(train_leng+vali_leng),vali_leng)';
 validPer1=ER;
 trainPer1=ER_Train;
 
 
 
 
 
 %{
      ER=sqrt(2*error/(size(Data_Vali,1)));
      if(Min_Error>ER)
          Min_Error=ER;
          Min_M=M;
          Min_lamb=lamb;
      end
 
 %}
 
 
 
%    end
% end
 
%{ 
M=Min_M;
lamb=Min_lamb;


Init_Centers=kMeansInitCentroids(Data_Train, M-1);
[Centers, Memberships]=kMeans(Data_Train, Init_Centers, max_iters);

Sigm_Gaus=Sigm_Gaussian(Data_Train, Centers, Memberships);
for q=1:1:M-1
    for p=1:1:D
       if(abs(Sigm_Gaus(p,p,q))<0.05)
           Sigm_Gaus(p,p,q)=0.2;
        end
    end
end





[NUM,TXT,RAW]=xlsread('Data.xls');
[NUM1,TXT1,RAW1]=xlsread('Data.xls');


leng=size(NUM,1);
leng1=size(NUM1,1);
train_leng=round(8/10*leng);
train_leng1=round(8/10*leng1);
vali_leng=round(1/10*leng);
vali_leng1=round(1/10*leng1);
test_leng=leng-train_leng-vali_leng;
max_iters=100;
Data_Train=NUM(1:train_leng,2:47);
Label_Train=NUM(1:train_leng,1);
Data_Vali=NUM(train_leng:(train_leng+vali_leng),2:47);
Label_Vali=NUM(train_leng:(train_leng+vali_leng),1);
Data_Test=NUM((train_leng+vali_leng):leng,2:47);
Label_Test=NUM((train_leng+vali_leng):leng,1);
D=size(Data_Train,2);




Phi=zeros(size(Data_Train,1),M-1);
One=ones(size(Data_Train,1),1);
Phi=[One,Phi];
 for i=1:1:size(Phi,1)
    for j=1:1:(size(Phi,2)-1)
    
       Phi(i,j+1)=Gaus_Dis(Data_Train(i,:),Centers(j,:),Sigm_Gaus(:,:,j));
       
   end
 end

 I=diag(D);
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
 
 
 M1=M;
 mu1=Centers';
 Sigma1=Sigm_Gaus;
 lambda1=Min_lamb;
 trainInd1=linspace(1,train_leng1,train_leng1)';
 validInd1=linspace((train_leng1+1),(train_leng1+vali_leng1),vali_leng1)';
 w1=W;
 validPer1=ER;
 trainPer1=ER_Train;
 %}






 
 