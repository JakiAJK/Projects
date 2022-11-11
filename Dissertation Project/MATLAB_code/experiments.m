 clc
clear all
close all
warning off


x=imread("cameraman.png");
x=im2gray(x);
%x=imbilatfilt(x,60,0.4);
imshow(x);
a=[1,180];
b=[3,39];
a_1=randi([a(1),a(end)]);
a_2=randi([b(1),b(end)]);
len=a_2;
th=a_1;
h=fspecial('motion', len, th);


k=imfilter(x,h,"symmetric",'same','conv');
%k=imread("Kaggle\motion_blurred\3_HUAWEI-NOVA-LITE_M.jpg");
%k=im2gray(k);
figure;
imshow(uint8(k));


for klm=a(1):a(end)
    hn=fspecial("motion",len,klm);
    k1=edgetaper(k,hn);
    j=deconvlucy(k1,hn,10);
    jjj(klm)=psnr(uint8(j),x);
    jfp(klm)=ssim(uint8(j),x);
end
figure;
jjj(jjj==0)=nan;
jfp(jfp==0)=nan;
plot(jjj);
%figure;
%plot(jfp);
[valu,pos] = max(jjj);
jjj=-(valu-jjj)*100/valu;
avb=a(1):a(end);
avb=abs(pos-avb);%*100/length(avb);

lio=b(1):b(end);
for klm=lio
    hn=fspecial("motion",klm,th);
    k1=edgetaper(k,hn);
    j=deconvlucy(k1,hn,10);
    jjk(klm)=psnr(uint8(j),x);
    jfo(klm)=ssim(uint8(j),x);
end
figure;
jjk(jjk==0)=nan;
jfo(jfo==0)=nan;
plot(jjk);
%figure;
%plot(jfo(lio));
[valu,pos] = max(jjk);
jjk=-(valu-jjk)*100/valu;
avb2=1:b(end);
avb2=abs(pos-avb2);%*100/length(avb2);
figure;
plot(avb,jjj);
%mean(jjj(avb==20))
figure;
plot(avb2,jjk);
jjk=jjj;
avb2=avb;
mean(jjk(avb2==1))
mean(jjk(avb2==2))
mean(jjk(avb2==3))
mean(jjk(avb2==5))
mean(jjk(avb2==10))
mean(jjk(avb2==20))
mean(jjk(avb2==30))
mean(jjk(avb2==50))
mean(jjk(avb2==70))
