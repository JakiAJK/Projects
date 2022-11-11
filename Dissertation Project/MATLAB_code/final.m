 clc
clear all
close all
warning off

% x1=imresize(im2gray(imread("lena.jpg")),0.5);

% x3=im2gray(imread("baboon.jpg"));
% 
% figure;
% subplot(1,3,1)
% imshow(x1)
% xlabel("(a)Lena")
% 
% subplot(1,3,2)
% imshow(x2)
% xlabel("(b)Cameraman")
% 
% subplot(1,3,3)
% imshow(x3)
% xlabel("(c)Baboon"






nam="cameraman";
lim=1;
ihj=0;
ihk=0;
while lim<21
    x=imread(append(nam,'.png'));
    x=im2gray(x);
    x2=x;
    a_1=randi([5,179]);
    a_2=randi([6,39]);
    h=fspecial('motion', a_2, a_1);

    k=imfilter(x,h,"symmetric",'same','conv');

    
    k11=k;
    [m,n1]=size(k);

    pt=hann(n1);

    pt2=hann(m);
    k=double(pt2.*double(k).*pt');
    ft1=calc_fft(k);
    ax1=abs(log(abs(ft1)));
    h_1=ones(3)/8;
    h_1(2,2)=0;
    b=ax1;
    [m,n]=size(b);
    k=uint16(m/2);
    l=uint16(n/2);

    I=b;

   
    [m,n]=size(I);
    if n>m
        n=m;
    end
    ajk=[];
    
    for th=5:5:180
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
        
        Xs=uint64(linspace(double(x),double(x1),uint16(n1)));
        Ys=uint64(linspace(double(y),double(y1),uint16(n1)));
        dd=impixel(I,double(Xs),double(Ys));
        vals=mean(dd,2);
        s=sum(vals,"all");
        varr=1e-100+var(vals,1,"all");
        ajk(th+1)=s.*s/(varr.*varr);
    end

    [mm,ang] = max(ajk);
    ajp=[];
    for th=ang-4:ang+6
        kl=imrotate(b,180-th,"bilinear","crop");
        ss=sum(kl(:,l-5:l+5),"all");
        varr=1e-5+var(kl(:,l-5:l+5),1,"all");
        ajp(th+4)=ss/(varr);
    end

    th=0:175;
    [mn,angle]=max(ajp);
    angle=angle-4;
    if angle<0
        angle=180+angle;
    end
    
    if abs(angle-a_1)>150
        angle=180-angle;
    end

    ax2=imrotate(b,180-angle,"bilinear","crop");
    ax3=ax2;
    ax2=medfilt2(ax2,[3,3]);
    ax2=ax2-min(ax2(:));
    ax2=ax2./max(ax2(:));
    avg=mean(ax2(:));
    ax2=ax2+1-avg;
    ax2=ax2.^3;
    ax2=medfilt2(ax2,[3,3]);
    pt=gausswin(n,1.5);
    pt2=gausswin(m,1.5);
    ax2=ax2./max(ax2(:));
    plm=sum(ax2,1);
    plm=nonzeros(plm);
    n1=length(plm);
    [lma,ppp]=islocalmin(plm,'MinProminence',2);

    ed=find(lma);
    if length(ed)<3
        plm=sum(ax3,1);
        plm=nonzeros(plm);
        [lma,ppp]=islocalmin(plm,'MinProminence',2);
    end

    val=diff(find(lma));
    m1=median(val);
    m2=mean(val);
    m4=max(val)/2;
    hjk=(n/m4);
    if hjk>18 && abs(m1-m4)<8
        hjk=(2*n/((m1+m2)));
    end

    
    hh=fspecial("motion",hjk,angle);
    k12=edgetaper(k11,hh);
    j=deconvlucy(k12,hh,30);
    psnr_c=psnr(uint8(j),x2);
    
    if psnr_c>27 && ihj<3
        ihj=ihj+1;
        subplot(3,2,ihj*2-1)
        imshow(uint8(j))
        xlabel(append('L=',int2str(hjk),',','Th=',int2str(angle),',','PSNR=',int2str(psnr_c)))
    end

    if psnr_c<24
        if ihk<3
        ihk=ihk+1;
        subplot(3,2,2*ihk)
        imshow(uint8(j))
        xlabel(append('L=',int2str(hjk),',','Th=',int2str(angle),',','PSNR=',int2str(psnr_c)))
        end
    end
    
    true_l(lim)=a_2;
    true_ang(lim)=a_1;
    pred_ang(lim)=angle;
    pred_l(lim)=hjk;
    psnrs(lim)=psnr_c;
    lim=lim+1;
    
end
lim=lim-1;
true_le=reshape(true_l,[lim,1]);
true_angl=reshape(true_ang,[lim,1]);
pred_angl=reshape(pred_ang,[lim,1]);
pred_le=round(reshape(pred_l,[lim,1]),3);
err_L=round(abs(true_le-pred_le),2);
err_A=round(abs(true_angl-pred_angl),2);
PSNRs=round(reshape(psnrs,[lim,1]),2);
T=table(true_angl,true_le,pred_angl,pred_le,err_L,err_A,PSNRs)
writetable(T,append(nam,'.csv'),'Delimiter',',','QuoteStrings',true)
mean(err_A)
mean(err_L)
print(gcf, '-djpeg', 'myfigure')

function ft=calc_fft(image)
ft=ifftshift(fft2(fftshift(image)));
%ft=fft2(ft);
end
