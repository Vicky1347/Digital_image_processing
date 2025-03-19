clc; 
clear; 
close all;

% Read input image
img = imread('road.jpg'); % Change 'image.jpg' to your image file
gray_img = double(rgb2gray(img)); % Convert to grayscale and double precision

%% Sobel Edge Detection (Without Built-in)
sobel_x = [-1 0 1; -2 0 2; -1 0 1]; % Sobel filter in x-direction
sobel_y = [-1 -2 -1; 0 0 0; 1 2 1]; % Sobel filter in y-direction

Gx = conv2(gray_img, sobel_x, 'same'); % Convolution with Sobel x
Gy = conv2(gray_img, sobel_y, 'same'); % Convolution with Sobel y

sobel_edges = sqrt(Gx.^2 + Gy.^2); % Gradient magnitude
sobel_edges = uint8(255 * (sobel_edges / max(sobel_edges(:)))); % Normalize to 255

%% Prewitt Edge Detection (Without Built-in)
prewitt_x = [-1 0 1; -1 0 1; -1 0 1]; % Prewitt filter in x-direction
prewitt_y = [-1 -1 -1; 0 0 0; 1 1 1]; % Prewitt filter in y-direction

Px = conv2(gray_img, prewitt_x, 'same'); % Convolution with Prewitt x
Py = conv2(gray_img, prewitt_y, 'same'); % Convolution with Prewitt y

prewitt_edges = sqrt(Px.^2 + Py.^2); % Gradient magnitude
prewitt_edges = uint8(255 * (prewitt_edges / max(prewitt_edges(:)))); % Normalize

%% Canny Edge Detection (Without Built-in)
% 1. Gaussian Smoothing
gaussian_filter = fspecial('gaussian', [5 5], 1); % 5x5 Gaussian filter
smoothed_img = conv2(gray_img, gaussian_filter, 'same');

% 2. Compute Gradient using Sobel
Gx = conv2(smoothed_img, sobel_x, 'same');
Gy = conv2(smoothed_img, sobel_y, 'same');

gradient_magnitude = sqrt(Gx.^2 + Gy.^2);
gradient_direction = atan2(Gy, Gx) * (180 / pi); % Convert to degrees

% 3. Non-Maximum Suppression
[row, col] = size(gradient_magnitude);
nms_img = zeros(row, col); % Initialize with zeros

for i = 2:row-1
    for j = 2:col-1
        angle = gradient_direction(i, j);
        
        % Quantizing angles to 0, 45, 90, and 135 degrees
        if ((angle >= -22.5 && angle <= 22.5) || (angle >= 157.5 || angle <= -157.5))
            neighbors = [gradient_magnitude(i, j-1), gradient_magnitude(i, j+1)];
        elseif ((angle >= 22.5 && angle <= 67.5) || (angle >= -157.5 && angle <= -112.5))
            neighbors = [gradient_magnitude(i-1, j+1), gradient_magnitude(i+1, j-1)];
        elseif ((angle >= 67.5 && angle <= 112.5) || (angle >= -112.5 && angle <= -67.5))
            neighbors = [gradient_magnitude(i-1, j), gradient_magnitude(i+1, j)];
        else
            neighbors = [gradient_magnitude(i-1, j-1), gradient_magnitude(i+1, j+1)];
        end
        
        % Suppress non-maximum values
        if gradient_magnitude(i, j) >= max(neighbors)
            nms_img(i, j) = gradient_magnitude(i, j);
        else
            nms_img(i, j) = 0;
        end
    end
end

% 4. Double Thresholding and Edge Linking
low_threshold = 0.1 * max(nms_img(:));
high_threshold = 0.3 * max(nms_img(:));

strong_edges = nms_img >= high_threshold;
weak_edges = (nms_img < high_threshold) & (nms_img >= low_threshold);

% Edge Tracking by Hysteresis
final_edges = strong_edges;
for i = 2:row-1
    for j = 2:col-1
        if weak_edges(i, j)
            if any(any(strong_edges(i-1:i+1, j-1:j+1)))
                final_edges(i, j) = 1;
            end
        end
    end
end

final_edges = uint8(final_edges * 255); % Convert to 0-255

%% Display Results
figure;
subplot(2,2,1);
imshow(uint8(gray_img));
title('Original Image');

subplot(2,2,2);
imshow(sobel_edges);
title('Sobel Edge Detection');

subplot(2,2,3);
imshow(prewitt_edges);
title('Prewitt Edge Detection');

subplot(2,2,4);
imshow(final_edges);
title('Canny Edge Detection (Without Built-in)');
