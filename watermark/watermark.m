clc; clear; close all;

% Read the original image (host image)
host_img = im2double(imread('lena.png')); % Change to your host image file
watermark_img = im2double(imread('watermark.png')); % Change to your watermark file

% Resize watermark to match the LL sub-band size
[LL, LH, HL, HH] = dwt2(host_img, 'haar'); % Perform DWT
watermark_img = imresize(watermark_img, size(LL));

% Embedding Strength Factor (alpha)
alpha = 0.1;  % You can tune this parameter

% Embed the watermark in the LL sub-band
LL_watermarked = LL + alpha * watermark_img;

% Perform IDWT (Inverse DWT) to get the watermarked image
watermarked_img = idwt2(LL_watermarked, LH, HL, HH, 'haar');

% Display Images
figure;
subplot(1,3,1); imshow(host_img); title('Original Image');
subplot(1,3,2); imshow(watermark_img); title('Watermark');
subplot(1,3,3); imshow(watermarked_img); title('Watermarked Image');

% Save the watermarked image
imwrite(watermarked_img, 'watermarked_image.png');
