 clc
clear all
close all
warning off

lma=111;
sdf=zeros(uint16(1.1*lma));
rad=double(int16(lma/2));
ii=0;
for th=0:2:90
    x=double(int16(lma/2+rad*sind(th)));
    y=double(int16(lma-rad*cosd(th)));
    sdf(x,y)=1;
    if ii==0
        x1=x;
        y1=y;
        ii=1;
    end
    n1=1000;
    Xs=(uint64(linspace(double(x),double(x1),uint16(n1))));
    Ys=(uint64(linspace(double(y),double(y1),uint16(n1))));
    idx = sub2ind(size(sdf), double(Xs),double(Ys));
    sdf(idx)=1;
    x1=x;
    y1=y;
end
sdf=sdf./sum(sdf,"all");
% Remove zero rows
data=sdf;
data( all(~data,2), : ) = [];
% Remove zero columns
data( :, all(~data,1) ) = [];
sdf=data;
[m,n]=size(sdf);
sz=double(uint8((n-m)/2)-1);
sdf=padarray(sdf,[sz,0],0,'post');
sdf=padarray(sdf,[sz,0],0,'pre');
h=fspecial("motion",2*lma,45);
figure;
subplot(1,2,1);
imshow(sdf,[]);
subplot(1,2,2);
imshow(h,[]);

x=imread("cameraman.png");
x=im2gray(x);
%x2=x;
%x=imbilatfilt(x,60,0.4);
figure;
imshow(x);

figure;
imshow(abs(log(abs(calc_fft(x)))),[]);

kt=imfilter(x,h,"symmetric",'same','conv');

k=imfilter(x,sdf,"symmetric",'same','conv');
figure;
imshow([kt,k])

[m,n1]=size(k);
pt=hann(n1);
pt2=hann(m);
%k=double(pt2.*double(k).*pt');
%kt=double(pt2.*double(kt).*pt');

ft2=calc_fft(kt);
%figure;
ax2=abs(log(abs(ft2)));
%ax1=ax1(2*m/5:3*m/5,2*n1/5:3*n1/5);

ft1=calc_fft(k);
figure;
ax1=abs(log(abs(ft1)));
imshow([ax2,ax1],[]);

h1=ones(size(sdf));
[j,p1]=deconvblind(k,h1);
psnr(uint8(j),x)
ssim(uint8(j),x)
norm(double(j-k),'fro')^2/numel(k)


h1=ones(size(h));
[j2,p2]=deconvblind(kt,h1);
psnr(uint8(j2),x)
ssim(uint8(j2),x)
norm(double(j2-kt),'fro')^2/numel(kt)


figure;
imshow(uint8([j2,j]),[]);

j=deconvlucy(k,sdf,50);
psnr(uint8(j),x)
ssim(uint8(j),x)
norm(double(j-k),'fro')^2/numel(k)


j2=deconvlucy(kt,h,50);
psnr(uint8(j2),x)
ssim(uint8(j2),x)
norm(double(j2-kt),'fro')^2/numel(kt)


figure;
imshow(uint8([j2,j]),[]);

figure;
subplot(1,2,1);
imshow(p2,[]);
subplot(1,2,2);
imshow(p1,[]);




function ft=calc_fft(image)
ft=ifftshift(fft2(fftshift(image)));
%ft=fft2(ft);
end