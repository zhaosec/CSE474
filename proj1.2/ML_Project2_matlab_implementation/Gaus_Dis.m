function Gaus_Dis = Gaus_Dis(Data, Center, Sigm)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    D=size(Center,2);
    Gaus_Dis=exp((Data-Center)*inv(Sigm)*(Data-Center)'.*(-1/2));%/sqrt((2*pi)^D*det(Sigm));

end

