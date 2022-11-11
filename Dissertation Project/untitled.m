 clc
clear all
close all
warning off


x=imread("lena.jpg");
x=im2gray(x);

imshow(x);
th=90;
h=fspecial('motion',15, th);
%x=imbilatfilt(x);
k=imfilter(x,h,"symmetric",'same','conv');

figure;
imshow(uint8(k));

for klm=-90:90
    hn=fspecial("motion",15,klm+90);
    k1=edgetaper(k,hn);
    j=deconvlucy(k1,hn,10);
    jjj(klm+91)=psnr(uint8(j),x);
end
figure;
plot(jjj,0:180);


for klm=3:2:27
    hn=fspecial("motion",klm,90);
    k1=edgetaper(k,hn);
    j=deconvlucy(k1,hn,10);
    jjk(klm)=psnr(uint8(j),x);
end
figure;
plot(jjk);

ft1=calc_fft(k);
figure;
ax1=abs(log(abs(ft1)));
ax1=imbilatfilt(ax1);
%ax1=imbinarize(ax1);
[m,n]=size(ax1);
pt=gausswin(n,1);
ax1=pt.*ax1;
ax=(ax1./max(ax1));
imshow(ax,[])
%ax=ax1;


kernel1 = -1 * ones(3)/9;
kernel1(2,2) = 10/9;
ax=imfilter(ax,kernel1,"symmetric","same");
ax=(ax./max(ax));
ax(ax<0.65)=0;
%ax(ax<0.65)=0;
%ax(ax>0.6)=0.8;
%ax=medfilt2(ax,[3 3]);
ax(ax>0.8)=1;


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
ax=ax(:,1*temp:4*temp);
imshow(ax);



figure;
theta = 1:179;
[R,xp] = radon(ax,theta);
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

z=diff(k,1,2);
zz=xcorr2(z,z);
zs=sum(zz,1);
%figure;
%plot(zs);


for cc=3:2:29
    h11=fspecial("motion",cc,ang);
    k1=edgetaper(k,h11);
    j=deconvlucy(k1,h11,10);
    ll(cc)=psnr(uint8(j),x);
end
figure;
plot(ll);
[M,Ii] = max(ll);
Ii

h3=fspecial("motion",19,ang);
k=edgetaper(k,h3);
j=deconvlucy(k,h3,20);
psnr(uint8(j),x)
figure;
imshow(uint8(j));
%h2=ones(5);
%k=edgetaper(k,h2);

j=deconvblind(k,h3,20);
psnr(uint8(j),x)
%figure;
%imshow(uint8(j));

function ft=calc_fft(image)
ft=ifftshift(fft2(fftshift(image)));
%ft=fft2(ft);
end