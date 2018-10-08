function [mean_column] = mean_column(A)
%Calculate the mean of the columns of a matrix;
[row,column]=size(A);
mean_column=zeros(1,column);

for j=1:1:column
    temp=0;
    for i=1:1:row
        temp=temp+A(i,j);
    end
    
    mean_column(j)=temp/row;
end

    

end

