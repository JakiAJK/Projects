 clc
clear all
close all
warning off

x=imread("lena.jpg");
x=im2gray(x);
%x2=x;
figure;
imshow(x,[])
x=imbilatfilt(x,60,0.4);
[m,n1]=size(x);
pt=gausswin(n1,0.1);
pt2=gausswin(m,0.1);
%x2=uint8(pt2.*double(x2).*pt');
imshow(x);
th=45;
h=fspecial('motion',25, th);
figure;
imshow(h,[]);
%imsave()
k=imfilter(x,h,"symmetric",'same','conv');



%k=k(m/4:3*m/4,n1/4:3*n1/4);
%k=imread("Kaggle\motion_blurred\3_HUAWEI-NOVA-LITE_M.jpg");
%k=im2gray(k);
%k=k(m/5:4*m/5,n1/5:4*n1/5);
[m,n1]=size(k);
pt=gausswin(n1,1.5);
pt=hann(n1);
pt2=gausswin(m,1.5);
pt2=hann(m);
k=double(pt2.*double(k).*pt');
figure;

%k=k-min(k(:));
imshow(k,[]);
%imsave()
ft1=calc_fft(k);
figure;
ax1=abs(log(abs(ft1)));
imshow(ax1,[]);
%ax1=medfilt2(ax1,[3 3]);
%ax1=imbilatfilt(ax1,10,1);
[m,n1]=size(ax1);
pt=gausswin(n1,1);
pt2=gausswin(m,1);
%ax1=(pt2.*ax1.*pt');

h_1=ones(3)/8;
h_1(2,2)=0;
ax1=(ax1./max(ax1));

normA = ax1 - min(ax1(:));
%ax1 = normA ./ max(normA(:));
%ax=edge(ax,"sobel");
%ax=ax1;
%ax1=imfilter(ax1,h_1,"replicate","same","conv");
%ax=imbinarize(ax1,"global");
pp=ax1;%histeq(double(ax));
figure;
imshow(ax1,[])
b=pp;
%b=imresize(ax1,0.4);
figure;
imshow(pp,[]);
%plot(sum(b,1));
[m,n]=size(b);
k=uint8(m/2);
l=uint8(n/2);
%b(:,l-5:l+5)=0.1;
%b=edge(b,"sobel");
var(b(:,l-5:l+5),1,"all")
%improfile()
imshow(b,[]);

figure;
theta = 1:180;
[R,xp] = radon(b,theta);
imagesc(theta,xp,R);
title('R_{\theta} (X\prime)');
xlabel('\theta (degrees)');
ylabel('X\prime');
set(gca,'XTick',0:20:180);
colormap(hot);
colorbar

ft2=fft(R,[],1);
[m,n]=size(ft2);
azs=sum(abs(ft2(:,:)),1);
%azs=sum(R(m/2-2:m/2+2,:),1);
%azs=sum(R,1);
figure;
plot(azs);

for theta=1:9:180
    kl=imrotate(b,180-theta,"bilinear","crop");
    ss=sum(kl(:,l-5:l+5),"all");
    varr=1e-5+var(kl(:,l-5:l+5),1,"all");
    ajk(theta)=ss/(varr);
end
figure;
plot(ajk);
[mm,ang] = max(ajk);

for th=ang-9:ang+9
    kl=imrotate(b,180-th,"bilinear","crop");
    ss=sum(kl(:,l-5:l+5),"all");
    varr=1e-5+var(kl(:,l-5:l+5),1,"all");
    ajp(th)=ss/(varr);
end
[mn,angle]=max(ajp);
angle
   










function ft=calc_fft(image)
ft=ifftshift(fft2(fftshift(image)));
%ft=fft2(ft);
end
