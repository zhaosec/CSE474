[NUM,TXT,RAW]=xlsread('Data_Small_Scale.xls');
%[NUM,TXT,RAW]=xlsread('Data_Middle_Scale.xls');
%[NUM,TXT,RAW]=xlsread('Data.xls');
Min_Error=10000;

count=zeros(1,100);
pp=1;

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


%{
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


One=ones(size(Data_Train,1),1);

W=rand(1,(M));
%W=[0.2631,-0.0001,-0.0001,-0.0001,-0.0001,-0.0001,-0.0001];
Former_W=W;


for p=1:1:100000
    i=mod(p,train_leng)+1;
    I_rate=0.5/sqrt(i);
    Former_W=W;
    Gaus_kern=Gaus_Kern(Data_Train(i,:),Centers,Sigm_Gaus);
    
    W=Former_W-([1,Gaus_kern].*(Cal_Y(Data_Train(i,:),Former_W',Centers,Sigm_Gaus)-Label_Train(i))+Former_W.*lamb).*I_rate;
    Diff=W-Former_W;
    Abs_Diff=Diff*Diff';
    if(100000*Abs_Diff<0.00001)
         break;
    end
    count(pp)=count(pp)+1;
end

pp=pp+1;


 error=0;
 Y_out=zeros(1,size(Data_Vali,1));
 for p=1:1:size(Data_Vali,1)
     
     Data_Temp=[1,Data_Vali(p,:)];

     Y_out(p)=round(Cal_Y(Data_Vali(p),W',Centers,Sigm_Gaus));
     
     error=error+0.5*(Y_out(p)-Label_Vali(p))^2;
     
     
 end

 
 
ER=sqrt(2*error/(size(Data_Vali,1)));
 
     if(Min_Error>ER)
          Min_Error=ER;
          
          
      end
 

%}

    




[NUM,TXT,RAW]=xlsread('Data.xls');
[NUM1,TXT1,RAW1]=xlsread('Data.xls');

leng=size(NUM,1);
leng1=size(NUM1,1);
train_leng=round(8/10*leng);
train_leng1=round(8/10*leng1);
vali_leng=round(1/10*leng);
vali_leng1=round(1/10*leng1);
test_leng=leng-train_leng-vali_leng;
test_leng1=leng1-train_leng1-vali_leng1;
max_iters=100;
Data_Train=NUM(1:train_leng,2:47);
Label_Train=NUM(1:train_leng,1);
Data_Vali=NUM(train_leng:(train_leng+vali_leng),2:47);
Data_Vali1=NUM1(train_leng1:(train_leng1+vali_leng1),2:47);
Label_Vali=NUM(train_leng:(train_leng+vali_leng),1);
Label_Vali1=NUM1(train_leng1:(train_leng1+vali_leng1),1);
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


One=ones(size(Data_Train,1),1);

W=rand(1,M);
w01=W';
dw1=zeros(M,80000);
eta1=zeros(1,80000);
%W=[0.2631,-0.0001,-0.0001,-0.0001,-0.0001,-0.0001,-0.0001];
Former_W=W;


for p=1:1:80000
    i=mod(p,train_leng)+1;
    I_rate=0.5/sqrt(i);
    Former_W=W;
    Gaus_kern=Gaus_Kern(Data_Train(i,:),Centers,Sigm_Gaus);
    
    W=Former_W-([1,Gaus_kern].*(Cal_Y(Data_Train(i,:),Former_W',Centers,Sigm_Gaus)-Label_Train(i))+Former_W.*lamb).*I_rate;
    Diff=W-Former_W;
    dw1(:,p)=Diff';
    eta1(:,p)=I_rate;
    % Abs_Diff=Diff*Diff';
    % if(Abs_Diff<0.00001)
     %    break;
    % end
    count(pp)=count(pp)+1;
end

%{
cc=0;
for i=1:1:80000
    if(eta(1,i)~=0)
       cc=cc+1;
    end
end

dw1=zeros(M,cc);
eta1=zeros(1,cc);
for i=1:1:cc
    dw1(:,i)=dw(:,i);
    eta1(:,i)=eta(:,i);
end

%}
pp=pp+1;


 error=0;
 Y_out=zeros(1,size(Data_Vali1,1));
 for p=1:1:size(Data_Vali1,1)
     
     Data_Temp=[1,Data_Vali1(p,:)];

     Y_out(p)=Cal_Y(Data_Vali1(p),W',Centers,Sigm_Gaus);
     
     error=error+0.5*(Y_out(p)-Label_Vali1(p))^2;
     
     
 end

 
 
ER=sqrt(2*error/(size(Data_Vali1,1)));





