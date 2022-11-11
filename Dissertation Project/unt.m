 clc
clear all
close all
warning off


x=imread("lena.jpg");
x=im2gray(x);
%x2=x;
x=imbilatfilt(x,60,0.4);
[m,n1]=size(x);
pt=gausswin(n1,0.1);
pt2=gausswin(m,0.1);
%x2=uint8(pt2.*double(x2).*pt');
imshow(x);
th=90;
h=fspecial('motion',9, th);

k=imfilter(x,h,"symmetric",'same','conv');
k=imread("Kaggle\motion_blurred\3_HUAWEI-NOVA-LITE_M.jpg");
k=im2gray(k);
%k=k(m/5:4*m/5,n1/5:4*n1/5);
[m,n1]=size(k);
pt=gausswin(n1,1.5);
%pt=hann(n1);
pt2=gausswin(m,1.5);
%pt2=hann(m);
k=uint8(pt2.*double(k).*pt');
figure;
imshow(uint8(k));


ft1=calc_fft(k);
figure;
ax1=abs(log(abs(ft1)));
ax1=imbilatfilt(ax1,10,1);
%ax1=medfilt2(ax1,[3 3]);
%ax1=imbinarize(ax1);
[m,n1]=size(ax1);
pt=gausswin(n1,1);
pt2=gausswin(m,1);
ax1=(pt2.*ax1.*pt');


ax=(ax1./max(ax1));
imshow(ax1,[])
%ax=ax1;
%figure;

%plot(lma);

kernel1 = -1 * ones(3)/9;
kernel1(2,2) = 10/9;
ax=imfilter(ax,kernel1,"symmetric","same");
ax=(ax./max(ax));
ax(ax<0.65)=0;
%ax(ax>0.65)=1;
%ax(ax>0.6)=0.8;
%ax=medfilt2(ax,[3 3]);
ax(ax>0.85)=1;


%ax=imfilter(ax, [-1 0 1]');
%ax(ax<0)=0;
%ax=(ax./max(ax));
%ax(ax<0.4)=0;
%ax=medfilt2(ax,[3 3]);
%ax(ax>0.6)=0.8;
%ax(ax>0.6)=0.2;

[m,n]=size(ax);

%figure;
%imshow(ax);
%ax=edge(ax,"sobel");
temp=int16(n/5);
te=m/2;
ax(0.95*te:1.05*te,:)=[];
figure;
ax3=ax(:,1*temp:4*temp);
imshow(ax3);



figure;
theta = 1:179;
[R,xp] = radon(ax3,theta);
imagesc(theta,xp,R);
title('R_{\theta} (X\prime)');
xlabel('\theta (degrees)');
ylabel('X\prime');
set(gca,'XTick',0:20:180);
colormap(hot);
colorbar

ft2=fft(R,[],1);
[m,n]=size(ft2);
azs=sum(abs(ft2(1:m,:)),1);
figure;
plot(azs);
[M,Iii] = max(azs);
ang=Iii
%((azs(th)-azs(1))/azs(th))

ax2=imrotate(ax1,180-ang,"bilinear","crop");
figure;
imshow(ax2,[]);
figure;
plm=sum(ax2,1);
plm=nonzeros(plm);
plot(plm);
n1=length(plm);
lma=islocalmin(plm);
lma=lma(n1/2:5*n1/6);
%plot(lma);
val=diff(find(lma));
median(val)
mean(val)
val=val(1)
hjk=int8(n1/(median(val)))



z=diff(k,1,2);
zz=xcorr2(z,z);
zs=sum(zz,1);
%figure;
%plot(zs);



function ft=calc_fft(image)
ft=ifftshift(fft2(fftshift(image)));
%ft=fft2(ft);
end
