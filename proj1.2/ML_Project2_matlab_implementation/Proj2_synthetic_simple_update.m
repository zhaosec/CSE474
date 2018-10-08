load('synthetic.mat');
NUM2=x';
NUM=[t,NUM2];

width=size(NUM,2);
M=13;
lamb=0.01;

max_iters=200;
leng=size(NUM,1);
train_leng=round(9/10*leng);
vali_leng=round(1/10*leng);
Data_Train=NUM(1:train_leng,2:width);
Label_Train=NUM(1:train_leng,1);
Data_Vali=NUM((train_leng+1):(train_leng+vali_leng),2:width);
Label_Vali=NUM((train_leng+1):(train_leng+vali_leng),1);
D=size(Data_Train,2);

Init_Centers=kMeansInitCentroids(Data_Train, M);
[Centers, Memberships]=kMeans(Data_Train, Init_Centers, max_iters);

Sigm_Gaus=Sigm_Gaussian(Data_Train, Centers, Memberships);



for q=1:1:M
    for p=1:1:D
        
        Sigm_Gaus(p,p,q)=Sigm_Gaus(p,p,q);
        
    end
end

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
 
 
 error1=0;
 
 for i=1:1:size(Data_Vali,1)
     
      
     Y=Cal_Y(Data_Vali(i,:),W,Centers,Sigm_Gaus);
     
     error1=error1+0.5*(Y-Label_Vali(i))^2;
     
     
 end
 
  E1=error1;
      
      ER_Vali=sqrt(2*E1/(size(Data_Vali,1)));
      validPer2=ER_Vali;
      
      
      error=0;
 for i=1:1:size(Data_Train,1)
     

     Y=Cal_Y(Data_Train(i,:),W,Centers,Sigm_Gaus);
     
     error=error+0.5*(Y-Label_Train(i))^2;
     
     
 end
 
 
 E=error;%+(W')*W*0.5*lamb;
 
      ER_Train=sqrt(2*E/(size(Data_Train,1)));
      trainPer2=ER_Train;
    
 
  w2=W;
 M2=M;
 mu2=Centers';
 Sigma2=Sigm_Gaus;
 
 lambda2=lamb;
 trainInd2=linspace(1,train_leng,train_leng)';
 validInd2=linspace((train_leng+1),(train_leng+vali_leng),vali_leng)';
 lamb_Iter=0.00005;
 
  W=rand(1,M);
  w02=W';
  Init_Diff=W-w2';
  dw2=zeros(M,80000);
  eta2=zeros(1,80000);
  Former_W=W;
  
  for p=1:1:80000
      i=mod(p,train_leng);
      if(i==0)
       i=i+train_leng; 
      end
      
      I_rate=10/sqrt(p);
      Former_W=W;
      Gaus_kern=Gaus_Kern(Data_Train(i,:),Centers,Sigm_Gaus);
      W=Former_W-(Gaus_kern.*(Cal_Y(Data_Train(i,:),Former_W',Centers,Sigm_Gaus)-Label_Train(i,1))+Former_W.*lamb_Iter).*I_rate;
      Diff=W-Former_W;
      dw2(:,p)=Diff';
      eta2(:,p)=I_rate;
      if((W-w2')*(W-w2')'<(Init_Diff)*Init_Diff'*0.1&&p>2*train_leng)
        break;
      end
      



  end

  
  
