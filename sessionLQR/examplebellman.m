% Solution example bellman optimality
% IDC 2025

close all;
clearvars;

A=[2 1;-1 1];B=[0;1];x0=[2;-3];P=zeros(2,2,10);K=zeros(1,2,9);
x=zeros(2,1,11);x(:,:,1)=x0;
P(:,:,10)=[5 0;0 5];R=2;Q=[2 0;0 0.1];
t=0:1:10;

for i=9:-1:1
    K(:,:,i)=(R+B'*P(:,:,i+1)*B)\B'*P(:,:,i+1)*A;
    P(:,:,i)=(A-B*K(:,:,i))'*P(:,:,i+1)*(A-B*K(:,:,i))+...
        Q+K(:,:,i)'*R*K(:,:,i);
end
for i=1:9
    x(:,:,i+1)=A*x(:,:,i)-B*K(:,:,i)*x(:,:,i);
end

figure
plot(t,squeeze(x(2,1,:)))
hold on
plot(t,squeeze(x(1,1,:)))

figure
plot(t(1:end-2),squeeze(K(1,1,:)))
hold on
plot(t(1:end-2),squeeze(K(1,2,:)))

figure
plot(t(1:end-1),squeeze(P(1,1,:)))
hold on
plot(t(1:end-1),squeeze(P(2,2,:)))
hold on
plot(t(1:end-1),squeeze(P(1,2,:)))
for i=1:9
    x(:,:,i+1)=A*x(:,:,i)-B*K(:,:,i)*x(:,:,i+1);
end
