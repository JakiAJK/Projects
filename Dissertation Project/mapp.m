 clc
clear all
close all
warning off



% k=3;
% I=zeros(30);
% 
% for i=1:k
%     l=randi([20,30])
%     th=randi([1,179])
%     h=fspecial("motion",l,th);
%     h(30,30)=0;
%     figure;
%     imshow(h,[])
%end
x=imread("num_pll.png");
x=im2gray(x);
%x=imbilatfilt(x,60,0.4);
imshow(x);
th=20;
len=60;
h=fspecial('motion',len , th);
k=imfilter(x,h,"symmetric",'same','conv');
imshow(uint8(k));

k=edgetaper(k,h);

j=deconvlucy(k,h,30);
figure;
k=imnoise(k,"gaussian",0,1e-02);
imshow([x,uint8(k),uint8(j)])

% ft2=calc_fft(k);
% figure;
% imshow(abs(log(abs(ft2))),[])

% l=[8,15,30];
% th=[45,60,120];
% 
% for i=1:3
%     h=fspecial('motion',l(i),th(i));
%     k=imfilter(x,h,"symmetric",'same','conv');
%     subplot(2,3,i);
%     [m,n1]=size(k);
%     pt=hann(n1);
%     pt2=hann(m);
%     %k=double(pt2.*double(k).*pt');
%     imshow(uint8(k));
%     ft1=fft2(k);
%     ax=abs(log(abs(ft1)));
%     subplot(2,3,3+i);
%     imshow(ax,[]);
% end

function ft=calc_fft(image)
ft=ifftshift(fft2(fftshift(image)));
%ft=fft2(ft);
end