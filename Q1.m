%% Q1
clear,clc
%% a
load emaildata1_2020.mat;
figure(1)
plotdata(X,y);
title("Basic data");xlabel('first feature'), ylabel('second feature'),hold off
%% b
num_iters=10000;
alpha=0.1;
[m,n]=size(X);
X1=[ones(m,1),X];
[theta,J]=gradient_descent(X1,y,[0,0,0]',alpha,num_iters);
figure(2);
plotdata(X,y)
decision_boundary(X1,theta);
title("Basic data with linear decision boundary");xlabel('first feature'), ylabel('second feature'),hold off
%% c
m=length(y);
X2=[ones(m,1) X X(:,1).^2];
num_iters=20000;
alpha=0.1;
[thetaQ,Jq]=gradient_descent(X2,y,[0,0,0,0]',alpha,num_iters);
figure(3)
plotdata(X,y);
quadratic_boundary(X2,thetaQ);
title("Basic data with quadratic decision boundary");xlabel('first feature'), ylabel('second feature'),hold off
%% d
%    for alpha=0.1
figure(4);
subplot(211);plot(J); xlabel("Iter"); ylabel("Cost"); title("Linear with alpha=0.1");
subplot(212);plot(Jq); xlabel("Iter"); ylabel("Cost"); title("Quadratic alpha=0.1");
%   for alpha=0.01
[~,J]=gradient_descent(X1,y,[0,0,0]',0.01,num_iters);
[~,Jq]=gradient_descent(X2,y,[0,0,0,0]',0.01,num_iters);
figure(5);
subplot(211);plot(J); xlabel("Iter"); ylabel("Cost"); title("Linear with alpha=0.01");
subplot(212);plot(Jq); xlabel("Iter"); ylabel("Cost"); title("Quadratic alpha=0.01");
%   for alpha=0.001
[~,J]=gradient_descent(X1,y,[0,0,0]',0.001,num_iters);
[~,Jq]=gradient_descent(X2,y,[0,0,0,0]',0.001,num_iters);
figure(6);
subplot(211);plot(J); xlabel("Iter"); ylabel("Cost"); title("Linear with alpha=0.001");
subplot(212);plot(Jq); xlabel("Iter"); ylabel("Cost"); title("Quadratic alpha=0.001");
%   for alpha=0.0001
[~,J]=gradient_descent(X1,y,[0,0,0]',0.0001,num_iters);
[~,Jq]=gradient_descent(X2,y,[0,0,0,0]',0.0001,num_iters);
figure(7);
subplot(211);plot(J); xlabel("Iter"); ylabel("Cost"); title("Linear with alpha=0.00001");
subplot(212);plot(Jq); xlabel("Iter"); ylabel("Cost"); title("Quadratic alpha=0.00001");
%
% I check alpha=0.1,0.01,0.001,0.0001 and according to the graphs The best is a=0.1
%
%% e 
load email_test_data_2020_1.mat;
[m,n]=size(Xtest);
test=[ones(m,1),Xtest];   
figure(8)
subplot(211);plotdata(Xtest,ytest);decision_boundary(test,theta); title("Test group linear boundary");xlabel('first feature'), ylabel('second feature'); hold off;
test=[ones(m,1) Xtest Xtest(:,1).^2];
subplot(212);plotdata(Xtest,ytest); quadratic_boundary(test,thetaQ); title("Test group quadratic boundary");xlabel('first feature'), ylabel('second feature'); hold off;
%
%the Proper identification percentage in liner is:88% (22/25)
%the Proper identification percentage in quadratic is: 96% (24/25 
