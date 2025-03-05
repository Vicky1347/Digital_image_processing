image=imread('lila52.jpeg');
imshow(image)
gimg=rgb2gray(image);
imshow(gimg)

wavlet
% Get the size of the image
[M, N] = size(gimg);

% Ensure the dimensions are even
M = M - mod(M, 2);
N = N - mod(N, 2);
gimg = gimg(1:M, 1:N);

% Initialize matrices for sub-bands
LL = zeros(M/2, N/2);
LH = zeros(M/2, N/2);
HL = zeros(M/2, N/2);
HH = zeros(M/2, N/2);

% Perform the Haar wavelet transform
for i = 1:2:M
    for j = 1:2:N
        % Extract 2x2 block
        block = gimg(i:i+1, j:j+1);
        
        % Compute the coefficients
        LL((i+1)/2, (j+1)/2) = sum(block(:)) / 2;
        LH((i+1)/2, (j+1)/2) = (block(1,1) - block(1,2) + block(2,1) - block(2,2)) / 2;
        HL((i+1)/2, (j+1)/2) = (block(1,1) + block(1,2) - block(2,1) - block(2,2)) / 2;
        HH((i+1)/2, (j+1)/2) = (block(1,1) - block(1,2) - block(2,1) + block(2,2)) / 2;
    end
end

% Display the results
figure;
subplot(2,2,1), imshow(LL, []), title('Approximation (LL)');
subplot(2,2,2), imshow(LH, []), title('Horizontal Detail (LH)');
subplot(2,2,3), imshow(HL, []), title('Vertical Detail (HL)');
subplot(2,2,4), imshow(HH, []), title('Diagonal Detail (HH)');
