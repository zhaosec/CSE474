function [covariance_compute] = covariance_compute(A,B)
%Compute the convariance of two vectors;
a=length(A);
meanA=mean(A);
meanB=mean(B);
sum=0;

for i=1:1:a
    sum=sum+(A(i)-meanA)*(B(i)-meanB);
    
end

covariance_compute=sum/(a-1);

end

