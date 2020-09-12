%% Q2
clear;clc
%% a
load('emaildata2_2020.mat');
figure(1);
plotdata(X,y);title("Basic data");xlabel('first feature'); ylabel('second feature');hold off;
%% b
[m,n]=size(X);
X1=[ones(m,1),X];
alpha=0.1;
num_iters=20000;
[theta,J,grad]=gradient_descent(X1,y,[0,0,0]',alpha,num_iters);
figure(2);
plotdata(X,y);
decision_boundary(X1,theta);
title("Basic data with linear decision boundary");xlabel('first feature'); ylabel('second feature');hold off;
%
%in our case the Logistic Regression whit linear decision is not good for
% separate between span or normal mail
%
%% c
X=mapFeature(X(:,1),X(:,2));
[~,n]=size(X);
thetaP=zeros(n,1); 
lambda=0;
alpha=0.1; 
num_iters=20000;
[thetaP,JP,gradP]=gd_reg(X,y,thetaP,alpha,num_iters,lambda);
%% d
figure(3)
plotDecisionBoundary(thetaP, X, y); title("Basic data with polynomial decision boundary with lambda=0");
%% e
thetaC=zeros(n,1);
[theta]=gd_reg(X,y,thetaC,alpha,num_iters,1);
figure(4); plotDecisionBoundary(theta, X, y); title("Basic data with polynomial decision boundary with lambda=1");xlabel('first feature'); ylabel('second feature');hold off;
[theta]=gd_reg(X,y,thetaC,alpha,num_iters,10);
figure(5); plotDecisionBoundary(theta, X, y); title("Basic data with polynomial decision boundary with lambda=10");xlabel('first feature'); ylabel('second feature');hold off;
[theta]=gd_reg(X,y,thetaC,alpha,num_iters,100);
figure(6); plotDecisionBoundary(theta, X, y); title("Basic data with polynomial decision boundary withlambda=100");xlabel('first feature'); ylabel('second feature');hold off;
% 
% for lambda= 1, there is no big different from lamda=0 but we get some errors in the in identification ;
% for lambda=10 and 100 we can see that there are many errors in identification.
%
%% f
load emaildata3_2020.mat;
figure(7);
plotdata(X,y);title("Test group");xlabel('first feature'); ylabel('second feature');hold off;
X=mapFeature(X(:,1),X(:,2));
figure(8); plotDecisionBoundary(thetaP, X, y); title("Test data with polynomial decision boundary with lambda=0");xlabel('first feature'); ylabel('second feature');hold off;
%
% the Training data and the Test data Pretty similar so The prediction error will be small 
%
%% g
load emaildata2_2020.mat;
X=mapFeature(X(:,1),X(:,2));
theta0=zeros(n,1);
options = optimset ( 'GradObj' , 'on', 'MaxIter' , 5000 );
lambda=0;
figure(9)
[theta]= fminunc ( @(t) (costF_reg(t,X,y,lambda)), theta0,options ) ;
plotDecisionBoundary(theta, X, y); title("Basic data using 'fminunc' decision boundary with lambda=0");xlabel('first feature'); ylabel('second feature');hold off;
figure(10)
lambda=1;
[theta]= fminunc ( @(t) (costF_reg(t,X,y,lambda)), theta0,options) ;
plotDecisionBoundary(theta, X, y); title("Basic data using 'fminunc' decision boundary with lambda=1");xlabel('first feature'); ylabel('second feature');hold off;
figure(11)
lambda=10;
[theta]= fminunc ( @(t) (costF_reg(t,X,y,lambda)), theta0,options) ;
plotDecisionBoundary(theta, X, y); title("Basic data using 'fminunc' decision boundary with lambda=10");xlabel('first feature'); ylabel('second feature');hold off;
figure(12)
lambda=100;
[theta]= fminunc ( @(t) (costF_reg(t,X,y,lambda)), theta0,options ) ;
plotDecisionBoundary(theta, X, y); title("Basic data using 'fminunc' decision boundary with lambda=100");xlabel('first feature'); ylabel('second feature');hold off;
%
%for lambda=0 we get  Overfitting
%for lambda=1 we can see that there are no big different from lamda=1 in my gd_reg
%for lambda=10 we can see that there are many errors in detection like my gd_reg
%for lambda=100 we can see that there are many errors in detection but it smaller then my gd_reg

