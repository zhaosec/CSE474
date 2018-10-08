function Gaus_cons=Gaus_Cons(Sigm)

D=size(Sigm,2);
Gaus_cons=1/sqrt((2*pi)^D*det(Sigm));


end

