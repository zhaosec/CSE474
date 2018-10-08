load('synthetic.mat');
NUM2=x';
NUM=[t,NUM2];

width=size(NUM,2);
Min_Error=10000;

count=zeros(1,100);
pp=1;

M=17;
    lamb=0.1;

leng=size(NUM,1);

train_leng=round(8/10*leng);

vali_leng=round(1/10*leng);

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


One=ones(size(Data_Train,1),1);

W=rand(1,M);
w02=W';
dw2=zeros(M,10000);
eta2=zeros(1,10000);
%W=[0.2631,-0.0001,-0.0001,-0.0001,-0.0001,-0.0001,-0.0001];
Former_W=W;


for p=1:1:10000
    i=mod(p,train_leng);
    if(i==0)
       i=i+train_leng; 
    end
    I_rate=0.5/sqrt(i);
    Former_W=W;
    Gaus_kern=Gaus_Kern(Data_Train(i,:),Centers,Sigm_Gaus);
    
    W=Former_W-(Gaus_kern.*(Cal_Y(Data_Train(i,:),Former_W',Centers,Sigm_Gaus)-Label_Train(i))+Former_W.*lamb).*I_rate;
    Diff=W-Former_W;
    dw2(:,p)=Diff';
    eta2(:,p)=I_rate;
    % Abs_Diff=Diff*Diff';
    % if(Abs_Diff<0.00001)
     %    break;
    % end
    count(pp)=count(pp)+1;
end

%{
cc=0;
for i=1:1:10000
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
%{
pp=pp+1;


 error=0;
 Y_out=zeros(1,size(Data_Vali1,1));
 for p=1:1:size(Data_Vali1,1)
     
     Data_Temp=[1,Data_Vali1(p,:)];

     Y_out(p)=Cal_Y(Data_Vali1(p),W',Centers,Sigm_Gaus);
     
     error=error+0.5*(Y_out(p)-Label_Vali1(p))^2;
     
     
 end

 
 
ER=sqrt(2*error/(size(Data_Vali1,1)));
%}
