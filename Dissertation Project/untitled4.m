 clc
clear all
close all
warning off

x=imread("baboon.jpg");
x=im2gray(x);
%x2=x;
%x=imbilatfilt(x,60,0.4);
imshow(x);
a_1=101;%randi([10,170]);
a_2=16;%randi([3,39]);
h=fspecial('motion', a_2, a_1);

k=imfilter(x,h,"symmetric",'same','conv');

%k=imnoise(k,"gaussian",0,1e-04);
%k=k(m/4:3*m/4,n1/4:3*n1/4);
%k=imread("Kaggle\motion_blurred\3_HUAWEI-NOVA-LITE_M.jpg");
%k=im2gray(k);
k11=k;
%k=histeq(k);
[m,n1]=size(k);
%k=k(l/2:3*l/2,k2/2:3*k2/2);
%k=imresize(k,0.1);
%k=k(2*m/5:3*m/5,2*n1/5:3*n1/5);
k=imbilatfilt(k,100,0.7);
%k=x;
[m,n1]=size(k);

pt=hann(n1);

pt2=hann(m);
k=double(pt2.*double(k).*pt');
figure;

imshow(k,[]);
%k=x;
ft1=calc_fft(k);
figure;
ax1=abs(log(abs(ft1)));
%ax1=medfilt2(ax1,[2,2]);
%ax1=imbilatfilt(ax1,10,0.4);
%ax1=ax1(2*m/5:3*m/5,2*n1/5:3*n1/5);
imshow(ax1,[]);
h_1=ones(3)/8;
h_1(2,2)=0;
%ax1=(ax1./max(ax1));
%ax1=imfilter(ax1,h_1,"replicate","same","conv");
%ax1=histeq(ax1);
%ax1=double(imbinarize(ax1));
b=ax1;
%b=imresize(b,0.5);
figure;
[m,n]=size(b);
k=uint16(m/2);
l=uint16(n/2);
imshow(b,[]);

I=b;

%I=meshgrid(0:256);
[m,n]=size(I)
if n>m
    n=m;
end
%uint16(linspace(0,0,257))
for th=0:5:175
    x= (n/2*(1-sind(th)));
    if x==0 || th==180
        x=1;%abs(x-1);
    end
    y= (n/2*(1-cosd(th)));
    if y==0 || th==180
        y=1;
    end
    x1= abs(n-x);
    y1= abs(n-y);
    m1=abs(x1-x)+1;
    n1=abs(y1-y)+1;
    n1=1000;
    %if m1>n1
    %    n1=(m1);
    %end
    Xs=uint64(linspace(double(x),double(x1),uint16(n1)));
    Ys=uint64(linspace(double(y),double(y1),uint16(n1)));
    dd=impixel(I,double(Xs),double(Ys));
    %idx = sub2ind(size(I), double(Xs),double(Ys));
    %I(idx)=0;
    %figure;
    %imshow(I,[])
    %a1=a2
    vals=mean(dd,2);
    s=sum(vals,"all");
    varr=1e-100+var(vals,1,"all");
    ajk(th+1)=s.*s/(varr.*varr);
end

[mm,ang] = max(ajk);

for th=ang-6:ang+4
    kl=imrotate(b,180-th,"bilinear","crop");
    ss=sum(kl(:,l-5:l+5),"all");
    varr=1e-5+var(kl(:,l-5:l+5),1,"all");
    ajp(th+6)=ss/(varr);
end

th=0:175;
[mn,angle]=max(ajp);
angle=angle-6;

a1=th;
b1=ajk;
[b1_max,ang] = max(ajk);
a1_max = a1(ang);
figure;
plot(a1, b1, a1_max, b1_max, 'ro');
hold on
% plot([a1_max a1_max],[0 b1_max],'--m')
% plot([a1_max a1_max],[0 b1_max],'-.m')
plot([a1_max a1_max],[0 b1_max],':m')
title("Theta vs Statistic for angle detection")
xlabel('Angle of matrix values to the vertical axis') 
ylabel('sum/variance for the values along matrix') 


%figure;
%imshow(I,[]);
%figure;
%plot(ajk,'x')


ax2=imrotate(b,180-angle,"bilinear","crop");
figure;
imshow(ax2,[]);
%ax2=ax2./max(ax2(:));
ax3=ax2;
ax2=medfilt2(ax2,[6,6]);
ax2=ax2-min(ax2(:));
ax2=ax2./max(ax2(:));
avg=mean(ax2(:));
ax2=ax2+1-avg;
avg=mean(ax2(:));
ax2=ax2.^3;
%ax2=imfilter(ax2,h_1,"replicate","same","conv");
%ax2=double(imbinarize(ax2));
%ax2=histeq(ax2);
ax2=medfilt2(ax2,[3,3]);
%ax2=ax2.^3;
pt=gausswin(n,1.5);
pt2=gausswin(m,1.5);
ax2=(pt2.*ax2.*pt');
%ax2=ax2.*ax2;
ax2=ax2./max(ax2(:));
%ax2=histeq(ax2);
figure;
imshow(ax2,[]);
figure;
plm=sum(ax3,1);
plm=nonzeros(plm);
n1=length(plm);
[lma,ppp]=islocalmin(plm,'MinProminence',20);

%lma=lma(n1/5:4.*n1/5);

%a=t
plot(plm,'b');
title("Plot of collapsed 1D Fourier spectrum")
xlabel('Width of the image') 
ylabel('Intensity') 


ed=find(lma);
if length(ed)<3
    plm=sum(ax3,1);
    plm=nonzeros(plm);
    [lma,ppp]=islocalmin(plm,'MinProminence',20);
end
AAA=1:n1;
plot(AAA,plm,AAA(lma),plm(lma),'r*')
axis tight
val=diff(find(lma));
m1=median(val)
m2=mean(val)
m3=mean(val(end))
m4=max(val)/2;
val1=(4*m3+2*m1+m2)/7
%hjk=int8(n/((val1)))
hjk=int8(n/m4)
if hjk>18
    hjk=int8(2*n/((2*m1)))
end

ppp(lma)


function ft=calc_fft(image)
ft=ifftshift(fft2(fftshift(image)));
%ft=fft2(ft);
end
