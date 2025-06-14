n = 4;
T0 = 0.54;
K = 0.39;
Ti = 1.98;
Td = Ti/4;
Ts = Td/8;

s=poly(3,'s');
W1=2*(Td*s/(1+Ts*s)+1+1/(Ti*s))*K*(1/(1+T0*s)^n);
W=W1/(1+W1);
disp("W: ", W);

h=0.1;
sys = syslin('c', W);
sysd=dscr(sys,h);
A = sysd.A;

I=eye(A);
H=lyap(A,-I,'d');

l=spec(H);
disp('Собственные числа H:', l);
disp('Запас устойчивости:', norm(H, 2));
