 clc
clear all
close all
warning off


x=imread("cameraman.png");
x=im2gray(x);
x=imbilatfilt(x,60,0.4);
imshow(x);
th=70;
len=15;
h=fspecial('motion',len, th);

k=imfilter(x,h,"symmetric",'same','conv');
%k=imread("Kaggle\motion_blurred\3_HUAWEI-NOVA-LITE_M.jpg");
%k=im2gray(k);
figure;
imshow(uint8(k));



j=deconvreg(k,h);
psnr(uint8(j),x)
ssim(uint8(j),x)
figure;
imshow(uint8(j));

j=deconvlucy(k,h,10);
psnr(uint8(j),x)
ssim(uint8(j),x)
figure;
imshow(uint8(j));

j=deconvblind(k,h);
psnr(uint8(j),x)
ssim(uint8(j),x)
figure;
imshow(uint8(j));

j=deconvwnr(k,h);
psnr(uint8(j),x)
ssim(uint8(j),x)
figure;
imshow(uint8(j));


dd=1;
for t=1:5
    j=deconvlucy(k,h,t*10);
    val(dd)=psnr(uint8(j),x);
    dd=dd+1;
end

figure;
imshow(uint8(j));
val;
plot(val)

