 clc
clear all
close all
warning off

x=imread("lena.jpg");
x=im2gray(x);
%x2=x;
%x=imbilatfilt(x,60,0.4);
imshow(x);
th=152;
h=fspecial('motion',19, th);

k=imfilter(x,h,"symmetric",'same','conv');
%k=k(m/4:3*m/4,n1/4:3*n1/4);
%k=imread("Kaggle\motion_blurred\3_HUAWEI-NOVA-LITE_M.jpg");
%k=im2gray(k);
%k=histeq(k);
[m,n1]=size(k);
%k=k(l/2:3*l/2,k2/2:3*k2/2);
%k=imresize(k,0.1);
%k=k(2*m/5:3*m/5,2*n1/5:3*n1/5);
k=imbilatfilt(k,100,0.7);
[m,n1]=size(k);

pt=hann(n1);

pt2=hann(m);
k=double(pt2.*double(k).*pt');
figure;

imshow(k,[]);
ft1=calc_fft(k);
figure;
ax1=abs(log(abs(ft1)));
%ax1=ax1(2*m/5:3*m/5,2*n1/5:3*n1/5);
imshow(ax1,[]);
h_1=ones(3)/8;
h_1(2,2)=0;
%ax1=(ax1./max(ax1));
%ax1=imfilter(ax1,h_1,"replicate","same","conv");
%ax1=histeq(ax1);
%ax1=double(imbinarize(ax1));
b=ax1;
b=imresize(b,0.4);
figure;
[m,n]=size(b);
k=uint8(m/2);
l=uint8(n/2);
imshow(b,[]);

for theta=1:9:180
    kl=imrotate(b,180-theta,"bilinear","crop");
    ss=sum(kl(:,l-5:l+5),"all");
    varr=1e-5+var(kl(:,l-5:l+5),1,"all");
    ajk(theta)=ss/(varr);
    if length(ajk)>10 && ajk(end)<0.85*ajk(end-9)
        break
    end
end
figure;
plot(ajk);
[mm,ang] = max(ajk);

for th=ang-6:ang+4
    kl=imrotate(b,180-th,"bilinear","crop");
    ss=sum(kl(:,l-5:l+5),"all");
    varr=1e-5+var(kl(:,l-5:l+5),1,"all");
    ajp(th+6)=ss/(varr);
end
[mn,angle]=max(ajp);
angle=angle-6;
   
[m,n]=size(ax1);
ax1=(ax1./max(ax1));
ax2=imrotate(ax1,180-angle,"bilinear","crop");
ax2=ax2-min(ax2(:));
ax2=ax2./max(ax2(:));
%ax2=ax2.*ax2;
%ax2=imfilter(ax2,h_1,"replicate","same","conv");
%ax2=double(imbinarize(ax2));
%ax2=histeq(ax2);
ax2=medfilt2(ax2,[6,6]);
ax2=(pt2.*ax2.*pt');
ax2=ax2.*ax2;
pt=gausswin(n,1.5);
pt2=gausswin(m,1.5);
ax2=ax2./max(ax2(:));
figure;
imshow(ax2,[]);
figure;
plm=sum(ax2,1);
plm=nonzeros(plm);
plot(plm);
n1=length(plm);
lma=islocalmin(plm);
lma=lma(n1/5:n1/2-2);
%plot(lma);
val=diff(find(lma));
m1=median(val)
m2=mean(val)
m3=mean(val(end))
val1=(4*m3+2*m1+m2)/7
hjk=int8(n/((val1)))










function ft=calc_fft(image)
ft=ifftshift(fft2(fftshift(image)));
%ft=fft2(ft);
end
