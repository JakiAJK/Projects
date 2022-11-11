 clc
clear all
close all
warning off


x=imread("lena.jpg");
x=im2gray(x);
%x=imbilatfilt(x,60,0.4);
subplot(2,3,1);
imshow(x);
a=[0,179];
b=[3,13];
th=132;%randi([a(1),a(end)]);
len=7;%randi([b(1),b(end)]);
h=fspecial('motion', len , th);


k=imfilter(x,h,"symmetric",'same','conv');
k=imnoise(k,"gaussian",0,1e-04);
k11=k;
%k=imbilatfilt(k,50,0.1);
%k=medfilt2(k,[7,7]);
%k=imread("Kaggle\motion_blurred\3_HUAWEI-NOVA-LITE_M.jpg");
%k=im2gray(k);
subplot(2,3,2);
imshow(uint8(k));


algorithm=["Lucy_Richardson";"Regularisation";"Blind";"Wiener";"Inverse"];
k=edgetaper(k,h);

p=1;
j=deconvlucy(k,h,10);
psnrs(p)=psnr(uint8(j),x);
ssims(p)=ssim(uint8(j),x);
mses(p) = norm(double(j-k),'fro')^2/numel(k);
subplot(2,3,p+2);
imshow(uint8(j));


p=2;
j=deconvreg(k,h);
psnrs(p)=psnr(uint8(j),x);
ssims(p)=ssim(uint8(j),x);
mses(p) = norm(double(j-k),'fro')^2/numel(k);
subplot(2,3,p+2);
imshow(uint8(j));

p=3;
[m,n]=size(h);
h1=ones([m,n]);
j=deconvblind(k,h1);
psnrs(p)=psnr(uint8(j),x);
ssims(p)=ssim(uint8(j),x);
mses(p) = norm(double(j-k),'fro')^2/numel(k);
subplot(2,3,p+2);
imshow(uint8(j));

p=4;
j=deconvwnr(k,h);
psnrs(p)=psnr(uint8(j),x);
ssims(p)=ssim(uint8(j),x);
mses(p) = norm(double(j-k),'fro')^2/numel(k);
subplot(2,3,p+2);
imshow(uint8(j));

% p=5;
% [m,n]=size(k);
% [m1,n1]=size(h);
% k1=fft2(k);
% h1=calc_fft(h1);
% h1=(fft2(h,m,n));
% %j1=k1./h1;
% j = real(ifft2(fft2(k)./h1));
% 
% %j=real(ifft2(j1));
% %j=real(fftshift(ifft2(ifftshift(j1))));
% %j=deco(k,h);
% j=uint8(j);
% psnrs(p)=psnr(uint8(j),x);
% ssims(p)=ssim(uint8(j),x);
% mses(p) = norm(double(j-k),'fro')^2/numel(k);
% figure;
% imshow(uint8(j));


p=double(p);
PSNR=round(reshape(psnrs,[p,1]),2);
SSIM=round(reshape(ssims,[p,1]),2);
MSE=round(reshape(mses,[p,1]),2);
%psnrs=array2table(psnrs);
%ssims=array2table(ssims);
T=table(algorithm(1:p),PSNR,SSIM,MSE)
writetable(T,'myData.csv','Delimiter',',','QuoteStrings',true)
% 
for t=1:29
    if t>10
        t=t*10-90;
    end
    j=deconvlucy(k,h,t);
    val(t)=psnr(uint8(j),x);
end
val(val==0)=nan;
jjk=val;
figure;
[valu,pos] = max(jjk);
%jjk=-(valu-jjk)*100/valu;

plot(jjk,'r*')
title("PSNR vs Iterations")
xlabel('Number of Iterations') 
ylabel('PSNR of reconstructed image') 


j=deconvlucy(k,h,pos);
figure;
imshow([x,uint8(k),uint8(j)])

% kk=5;
% lls=[0,2,0,7,0];
% angs=[0,0,10,0,60];
% subplot(1,kk+1,1);
% imshow(uint8(k));
% for i=1:kk
%     h=fspecial("motion",len+lls(i),th-angs(i));
%     k=edgetaper(k11,h);
%     j=deconvlucy(k,h,30);
%     subplot(1,kk+1,i+1);
%     imshow(uint8(j));
%     pkj(i)=psnr(uint8(j),x);
% end
% jjk=pkj;
% [valu,pos] = max(jjk);
% jjk=-(valu-jjk)*100/valu;
% pkj
% jjk

function ft=calc_fft(image)
ft=ifftshift(fft2(fftshift(image)));
%ft=fft2(ft);
end

