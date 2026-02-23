clear all, close all, clc

termination = {'jan','feb','mar','apr','may','jun','jul','aug','sept','oct','nov','dec'};
for idx_month=1:12
    I = imread(['earth_' termination{idx_month} '.jpg']); %in int16

    % imshow(I)
    % 
    % figure
    % surf(I(1:10:end,1:10:end,1))
    % figure
    % surf(I(1:10:end,1:10:end,2))
    % figure
    % surf(I(1:10:end,1:10:end,3))
    
    %[find_ocean] = find(abs(I(:,:,1)-11)>25 | abs(I(:,:,2)-10)>25 | abs(I(:,:,3)-50)>25);
    
    I_red = I(:,:,1);
    I_green = I(:,:,2);
    I_blue = I(:,:,3);

    ocean_color = [2,5,19];
    ocean_gray = [235,235,235];
    threshold = 20;
    threshold_gray = 4;
    white_threshold = 10;

    % check_ocean = ((abs(I_red-ocean_color(1))<=threshold &...
    %                 abs(I_green-ocean_color(2))<=threshold &...
    %                 abs(I_blue-ocean_color(3))<=threshold) |...
    %                 (abs(I_red-ocean_gray(1))<=threshold_gray &...
    %                 abs(I_green-ocean_gray(2))<=threshold_gray &...
    %                 abs(I_blue-ocean_gray(3))<=threshold_gray));
    check_ocean = (abs(I_red-ocean_color(1))<=threshold &...
                   abs(I_green-ocean_color(2))<=threshold &...
                   abs(I_blue-ocean_color(3))<=threshold) & (I_blue>=1.5*I_red | (I_red<15 & I_green<15 & I_blue<15));
    
    [ocean_black] = find(check_ocean);
    [land_white]  = find(~check_ocean);
    % 
    % I_red(:) = 0;
    % I_green(:) = 0;
    % I_blue(:) = 0;
    % 
    I_red(ocean_black) = 0;
    I_green(ocean_black) = 0;
    I_blue(ocean_black) = 0;
    I_red(land_white) = 255;
    I_green(land_white) = 255;
    I_blue(land_white) = 255;
    
    I_new = I;
    
    I_new(:,:,1) = I_red; %I_red
    I_new(:,:,2) = I_green;
    I_new(:,:,3) = I_blue; %I_blue
    
    % figure
    % surf(I_new(1:10:end,1:10:end,1))
    % figure
    % surf(I_new(1:10:end,1:10:end,2))
    % figure
    % surf(I_new(1:10:end,1:10:end,3))
    
    %imshow(I_new)
    % foldername = 'ocean_black_maps';
    % filename = fullfile(foldername,sprintf('color_oceanblack_dec.jpg',i,j));
    imwrite(uint8(I_new),['color_oceanblack_' termination{idx_month} '.jpg']) %from float to uint16, 0 to 32768
    
    I_red = I(:,:,1);
    I_green = I(:,:,2);
    I_blue = I(:,:,3);
    I_red(ocean_black) = 255;
    I_green(ocean_black) = 255;
    I_blue(ocean_black) = 255;
    
    I_new = I;
    
    I_new(:,:,1) = I_red; %I_red
    I_new(:,:,2) = I_green;
    I_new(:,:,3) = I_blue; %I_blue
    imwrite(uint8(I_new),['earth_' termination{idx_month} '_adj.jpg']) %from float to uint16, 0 to 32768
    % imshow(I_new)
end
% Icloud = imread("cloud_combined_2048.tif"); %in int16

% figure
% surf(Icloud(1:10:end,1:10:end,1))
% figure
% surf(Icloud(1:10:end,1:10:end,2))
% figure
% surf(Icloud(1:10:end,1:10:end,3))
% 
% el_everest = 8848; %m
% 
% I_float = double(I)/255*el_everest; %from 0 to 8848
% max(max(I_float))
% min(min(I_float))
% 
% imwrite(uint16(I_float),["elevationEarth_scaled.tiff"]) %from float to uint16, 0 to 32768
% I1 = imread(["elevationEarth_scaled.tiff"]);
% figure(2)
% imshow(I1)
% max(max(I1))
% min(min(I1))
% 
% % fprintf("read\n")
% % downscale = 20;
% % 
% % h = size(I,1);
% % w = size(I,2);
% % 
% % h_scale = h/downscale;
% % w_scale = w/downscale;
% % 
% % elevation = zeros(h_scale,w_scale);
% % for i=1:h_scale
% %     Itemp = I((i-1)*downscale+1:i*downscale,:);
% %     if rem(i,100)==0
% %         fprintf("-- i: %i out of %i --\n",i,h_scale)
% %     end
% %     parfor j=1:w_scale
% % %         if rem(j,1)==0
% % %             fprintf("---- j: %i out of %i ----\n",j,w_scale)
% % %         end
% %         elevation(i,j) = mean(mean(Itemp(:,(j-1)*downscale+1:j*downscale))); 
% %     end
% % end
% % max(max(elevation))
% % min(min(elevation))
% 
% % imshow(uint16(elevation))
% 
% % imwrite(uint16(elevation+32768),["elevation_" + num2str(downscale) + ".tiff"]) %from int16 to uint16
% % I1 = imread(["elevation_" + num2str(downscale) + ".tiff"]);
% % max(max(I1))
% % min(min(I1))
% 
% % imwrite(uint16(elevation+32768),["elevationEarth.tiff"]) %from int16 to uint16
