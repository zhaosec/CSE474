function Sigm_Gaussian = Sigm_Gaussian(Data, Centers, memberships)

[row1,column1]=size(Data);
[row2,column2]=size(Centers);

Sigm_Gaussian=zeros(column1,column1,row2);
Temp_Data=ones(size(Data,1),size(Data,2));

for j=1:1:row2

        pointer=0;
        for k=1:1:row1
            if(memberships(k)==j)
               pointer=pointer+1;
               Temp_Data(pointer,:)=Data(k,:);
            end
            
        Var_Col=var_column(Temp_Data(1:pointer,:));
        
        for p=1:1:column1
            Sigm_Gaussian(p,p,j)=Var_Col(1,p);
        end
        
        end
        

    
end



end

