k=imread("Kaggle\motion_blurred\14_IPHONE-7_M.jpeg");
j=imread("Kaggle\sharp\14_IPHONE-7_S.jpeg");
k=im2gray(k);
j=im2gray(j);
%k=k(m/5:4*m/5,n1/5:4*n1/5);
[m,n1]=size(k);
pt=gausswin(n1,1.5);
pt=hann(n1);
pt2=gausswin(m,1.5);
pt2=hann(m);
%k=double(pt2.*double(k).*pt');
figure;

%k=k-min(k(:));
imshow(k,[]);
%imsave()
ft1=calc_fft(k);
figure;
ax1=abs(log(abs(ft1)));
imshow(ax1,[]);

ft2=calc_fft(j);
figure;
ax2=abs(log(abs(ft2)));
imshow(ax2,[]);

ghg=ft1.\ft2;
gf=real(ifft2(ghg));
figure;
imshow((gf),[])

function ft=calc_fft(image)
ft=ifftshift(fft2(fftshift(image)));
%ft=fft2(ft);
end
