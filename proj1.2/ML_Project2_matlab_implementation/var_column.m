function [var_column] = var_column(A)
%Calculate the variance of each column for the matrix;
    [row,column]=size(A);
    var_column=zeros(1,column);
    
    for j=1:1:column
        mean=mean_column(A(:,j));
        temp=0;
        for i=1:1:row
            temp=temp+(A(i,j)-mean)^2;
        end
        
        var_column(j)=temp/(row-1);
            
    end
    

end

