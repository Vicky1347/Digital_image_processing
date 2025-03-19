clc; 
clear; 
close all;

% Read input image
img = imread('highwayroad.jpeg'); % Change 'image.jpg' to your image file
gray_img = rgb2gray(img); % Convert to grayscale



% Apply Sobel edge detection
sobel_edges = edge(gray_img, 'sobel');

% Apply Canny edge detection
canny_edges = edge(gray_img, 'canny');

% Apply Prewitt edge detection
prewitt_edges = edge(gray_img, 'prewitt');

% Display results
figure;
subplot(2,2,1);
imshow(gray_img);
title('Original Image');

subplot(2,2,2);
imshow(sobel_edges);
title('Sobel Edge Detection');

subplot(2,2,3);
imshow(canny_edges);
title('Canny Edge Detection');

subplot(2,2,4);
imshow(prewitt_edges);
title('Prewitt Edge Detection');

