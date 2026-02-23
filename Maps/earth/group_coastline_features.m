clear all, close all, clc

%Load Earth map and cloud map
mapEarth = imread('earth_apr.jpg');
clouds = imread('clouds.jpg');
%Reduce resolution of cloud map (interpolation)
xpoints = [0:size(clouds,2)-1]/(size(clouds,2)-1);
ypoints = [0:size(clouds,1)-1]/(size(clouds,1)-1);
xpointsq = [0:size(mapEarth,2)-1]/(size(mapEarth,2)-1);
ypointsq = [0:size(mapEarth,1)-1]/(size(mapEarth,1)-1);

[xmesh,ymesh] = meshgrid(xpoints,ypoints);
[xmeshq,ymeshq] = meshgrid(xpointsq,ypointsq);
resize_clouds = uint8(interp2(xmesh,ymesh,double(clouds),xmeshq,ymeshq));

%Plot maps
set(0, 'DefaultFigureVisible', 'off');
fig1 = figure(1);
imshow(mapEarth)

fig2 = figure(2);
set(fig2,'Visible','off')
imshow(mapEarth)
hold on
hclouds = imshow(resize_clouds, 'InitialMag', 'fit');
set(hclouds, 'AlphaData', resize_clouds)

%Size of Earth map (cloud map too)
[numy,numx] = size(mapEarth,[1 2]);

% mapEarth_cloud = uint8(mapEarth*0);

% coast_features_all = readmatrix('capes_gulfs_bays_peninsulas_deltas_straits.csv');
% latlon = coast_features_all(:,13:14); %deg

% % [x,y] = mercator(latlon(:,2),latlon(:,1));
% x = latlon(:,2)*pi/180;
% y = latlon(:,1)*pi/180;

%Load coastlines
C = load('coastlines');
Corg = C;
C.coastlon(C.coastlon>180) = C.coastlon(C.coastlon>180)-360; %clip to -180 to 180 degrees
x = C.coastlon*pi/180; %convert to radians
y = C.coastlat*pi/180;

%Can slice coast points to reduce number

%Convert from angles to pixel coordinates
xpix = numx/2+0.5+x/pi*numx/2;
ypix = numy/2+0.5-y*2/pi*numy/2;

% %Plot coastlines
% hold on
% plot(xpix,ypix,'r.')
xypix = [xpix,ypix];

% point = [1334,952];
% points = [1334,952;...
%           1587,1049;...
%           1004,938];
%Points of interest in pixel coordinates
points = readmatrix('digitalized_points.csv');
% latlon = coast_features_all(:,13:14); %deg

% rng(1) %seed for random numbers
%Define different colors for lines
[rgb_num,rgb_name] = getcolors();
perm_rgb = randi(size(rgb_num,1),size(points,1),1);

%Define marker size for plots
size_markers = [10,15,20];
size_markers_perm = size_markers(randi(3,size(points,1),1));

%Define thresholds for boxes
dist_threshold = 200; %px, size of circles around points
diff_threshold = 50; %px
coverage_threshold = 0.6; %per one
% num_clusters = 3;
bounding_box = zeros(size(points,1)*9,6);
% points_list = cell(size(points,1),1);
% box_list = cell(size(points,1),1);
% [xmeshclouds,ymeshclouds] = meshgrid(1:size(mapEarth,2),1:size(mapEarth,1));
counter_points = 0;
for idx_point=1:size(points,1) %For each of the interest points along coastline
    point = points(idx_point,:); %get point coordinates
    dist_point = sqrt(sum((xypix-point).^2,2)); %for all of the points on the coastline, find distance to interest point
    idx_select = find(dist_point<dist_threshold); %keep only nearby points
    xpix_select = xpix(idx_select); %index nearby points
    ypix_select = ypix(idx_select);
    
    % clusters = clusterdata([xpix_select,ypix_select],num_clusters);
    % mode_cluster = mode(clusters);
    % idx_cluster = find(clusters==mode_cluster);
    % hold on
    % scatter(xpix_select,ypix_select,10,clusters,'filled')
    color_plot = hex2rgb(['#' rgb_num{perm_rgb(idx_point),1} rgb_num{perm_rgb(idx_point),2} rgb_num{perm_rgb(idx_point),3}]);
    
    bounding_box_i = [min(xpix_select),max(xpix_select),min(ypix_select),max(ypix_select)]; %find bounds of selected points
    if idx_point>1 %if this is not the first box
        diff_bounding = sum(abs(bounding_box(1:counter_points,1:4)-bounding_box_i),2);
        if any(diff_bounding<diff_threshold) %if last box is similar to any of the previous
            bounding_box(counter_points+1,:) = []; %delete current box if similar to any of the previous
            diff_true = 0; %similar
        else
            diff_true = 1; %different
        end
    else
        diff_true = 1; %different
    end
    if diff_true %if box is different from all previous ones
        counter_points = counter_points+1;
        %Cloud cover
        roundx = round(xpix_select); %Check what pixel this coastline point corresponds to
        roundy = round(ypix_select);
        clouds_box = resize_clouds((roundx-1)*numy+roundy); %linear indexing using row and columns
        % mapEarth_cloud1 = mapEarth_cloud(:,:,1);
        % mapEarth_cloud1((roundx-1)*numy+roundy) = 255;
        % mapEarth_cloud(:,:,1) = uint8(mapEarth_cloud1);
        average_coverage = mean(clouds_box)/255 %mean of all points
        if average_coverage<coverage_threshold
            visible = 1;
        else
            visible = 0;
        end
        id_box = 1; %big box
        bounding_box(counter_points,:) = [bounding_box_i,id_box,visible];
        set(0, 'currentfigure', fig1)
        hold on
        % scatter(xpix_select,ypix_select,size_markers_perm(idx_point),color_plot,'filled','MarkerFaceAlpha',.5,'MarkerEdgeAlpha',.5);
        plot([bounding_box_i(1), bounding_box_i(1), bounding_box_i(2), bounding_box_i(2), bounding_box_i(1)],...
             [bounding_box_i(3), bounding_box_i(4), bounding_box_i(4), bounding_box_i(3), bounding_box_i(3)],'-','Color',color_plot);
        set(0, 'currentfigure', fig2)
        hold on
        % scatter(xpix_select,ypix_select,size_markers_perm(idx_point),color_plot,'filled','MarkerFaceAlpha',.5,'MarkerEdgeAlpha',.5);
        plot([bounding_box_i(1), bounding_box_i(1), bounding_box_i(2), bounding_box_i(2), bounding_box_i(1)],...
             [bounding_box_i(3), bounding_box_i(4), bounding_box_i(4), bounding_box_i(3), bounding_box_i(3)],'-','Color',color_plot);
        % hearthclouds = imshow(mapEarth_cloud, 'InitialMag', 'fit');
        % set(hearthclouds, 'AlphaData', mapEarth_cloud(:,:,1))
    end
    %Discretize box
    xhalf = 1/2*(bounding_box_i(1)+bounding_box_i(2));
    yhalf = 1/2*(bounding_box_i(3)+bounding_box_i(4));
    for idx_box=1:8
        switch idx_box
            case 1 %top half 
                id_box = 2; %half box
                find_subbox = ypix_select<=yhalf;
                miny = bounding_box_i(3);
                xselect_subbox = xpix_select(find_subbox);
                yselect_subbox = ypix_select(find_subbox);
                maxy = max(yselect_subbox);
                maxx = max(xselect_subbox);
                minx = min(xselect_subbox);
            case 2 %bottom half
                id_box = 2; %half box
                find_subbox = ypix_select>=yhalf;
                maxy = bounding_box_i(4);
                xselect_subbox = xpix_select(find_subbox);
                yselect_subbox = ypix_select(find_subbox);
                miny = min(yselect_subbox);
                maxx = max(xselect_subbox);
                minx = min(xselect_subbox);
            case 3 %right half
                id_box = 2; %half box
                find_subbox = xpix_select>=xhalf;
                maxx = bounding_box_i(2);
                xselect_subbox = xpix_select(find_subbox);
                yselect_subbox = ypix_select(find_subbox);
                minx = min(xselect_subbox);
                miny = min(yselect_subbox);
                maxy = max(yselect_subbox);
            case 4 %left half
                id_box = 2; %half box
                find_subbox = xpix_select<=xhalf;
                minx = bounding_box_i(1);
                xselect_subbox = xpix_select(find_subbox);
                yselect_subbox = ypix_select(find_subbox);
                maxx = max(xselect_subbox);
                miny = min(yselect_subbox);
                maxy = max(yselect_subbox);
            case 5 %top right
                id_box = 3; %quarter box
                find_subbox = ypix_select<=yhalf & xpix_select>=xhalf;
                xselect_subbox = xpix_select(find_subbox);
                yselect_subbox = ypix_select(find_subbox);
                maxy = max(yselect_subbox);
                miny = min(yselect_subbox);
                maxx = max(xselect_subbox);
                minx = min(xselect_subbox);
            case 6 %top left
                id_box = 3; %quarter box
                find_subbox = ypix_select<=yhalf & xpix_select<=xhalf;
                xselect_subbox = xpix_select(find_subbox);
                yselect_subbox = ypix_select(find_subbox);
                maxy = max(yselect_subbox);
                miny = min(yselect_subbox);
                maxx = max(xselect_subbox);
                minx = min(xselect_subbox);
            case 7 %bottom right
                id_box = 3; %quarter box
                find_subbox = ypix_select>=yhalf & xpix_select>=xhalf;
                xselect_subbox = xpix_select(find_subbox);
                yselect_subbox = ypix_select(find_subbox);
                maxy = max(yselect_subbox);
                miny = min(yselect_subbox);
                maxx = max(xselect_subbox);
                minx = min(xselect_subbox);
            case 8 %bottom left
                id_box = 3; %quarter box
                find_subbox = ypix_select>=yhalf & xpix_select<=xhalf;
                xselect_subbox = xpix_select(find_subbox);
                yselect_subbox = ypix_select(find_subbox);
                maxy = max(yselect_subbox);
                miny = min(yselect_subbox);
                maxx = max(xselect_subbox);
                minx = min(xselect_subbox);
        end
        if any(find_subbox) %if subbox not empty
            bounding_subbox_i = [minx,maxx,miny,maxy];
            diff_bounding = sum(abs(bounding_box(1:counter_points,1:4)-bounding_subbox_i),2);
            if any(diff_bounding<diff_threshold) %if last box is similar to any of the previous
                bounding_box(counter_points+1,:) = [];
                diff_true = 0;
            else
                diff_true = 1;
            end
        else
            diff_true = 0;
            bounding_box(counter_points+1,:) = [];
        end
        if diff_true
            counter_points = counter_points+1;%Cloud cover
            roundx = round(xselect_subbox);
            roundy = round(yselect_subbox);
            clouds_box = resize_clouds((roundx-1)*numy+roundy);
            % mapEarth_cloud1 = mapEarth_cloud(:,:,1);
            % mapEarth_cloud1((roundx-1)*numy+roundy) = 255;
            % mapEarth_cloud(:,:,1) = uint8(mapEarth_cloud1);
            average_coverage = mean(clouds_box)/255
            if average_coverage<coverage_threshold
                visible = 1;
            else
                visible = 0;
            end
            bounding_box(counter_points,:) = [bounding_subbox_i,id_box,visible];
            set(0, 'currentfigure', fig1)
            hold on
            % scatter(xpix_select,ypix_select,size_markers_perm(idx_point),color_plot,'filled','MarkerFaceAlpha',.5,'MarkerEdgeAlpha',.5);
            plot([bounding_subbox_i(1), bounding_subbox_i(1), bounding_subbox_i(2), bounding_subbox_i(2), bounding_subbox_i(1)],...
                 [bounding_subbox_i(3), bounding_subbox_i(4), bounding_subbox_i(4), bounding_subbox_i(3), bounding_subbox_i(3)],'-','Color',color_plot);
            set(0, 'currentfigure', fig2)
            hold on
            % scatter(xpix_select,ypix_select,size_markers_perm(idx_point),color_plot,'filled','MarkerFaceAlpha',.5,'MarkerEdgeAlpha',.5);
            plot([bounding_subbox_i(1), bounding_subbox_i(1), bounding_subbox_i(2), bounding_subbox_i(2), bounding_subbox_i(1)],...
                 [bounding_subbox_i(3), bounding_subbox_i(4), bounding_subbox_i(4), bounding_subbox_i(3), bounding_subbox_i(3)],'-','Color',color_plot);
            % hearthclouds = imshow(mapEarth_cloud, 'InitialMag', 'fit');
            % set(hearthclouds, 'AlphaData', mapEarth_cloud(:,:,1))
        end
    end
    % hold on
    % color_plot = hex2rgb(['#' rgb_num{perm_rgb(idx_point),1} rgb_num{perm_rgb(idx_point),2} rgb_num{perm_rgb(idx_point),3}]);
    % plot(points(idx_point,1),points(idx_point,2),'x','Color',color_plot,'LineWidth',1.5)
    % scatter(xpix_select(idx_cluster),ypix_select(idx_cluster),size_markers_perm(idx_point),color_plot,'filled','MarkerFaceAlpha',.5,'MarkerEdgeAlpha',.5)

end
set(fig1,'Visible','on')
set(fig2,'Visible','on')
% image1 = 'cloud_alpha_res_2048x1024_day_20_month_Oct_year_2024_hour_13_minute_36_second_48_PT.jpg';
% image2 = 'cloud_alpha_res_2048x1024_day_20_month_Oct_year_2024_hour_15_minute_36_second_58_PT.jpg';
% I1 = rgb2gray(imread(image1));
% I2 = rgb2gray(imread(image2));
% 
% Iint = (I1+I2)/2;
% figure
% subplot(2,2,1)
% imshow(I1)
% subplot(2,2,2)
% imshow(I2)
% subplot(2,2,3)
% imshow(Iint)