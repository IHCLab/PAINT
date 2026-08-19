function plot_result(AVIRIS, model, MSI, random_indices, num_random_pixels, cols, rows)
    %% Setting
    [H, W, ~] = size(model);
    MSI = imresize(MSI,2,"nearest");
    GT_2d = reshape(double(AVIRIS), [], size(AVIRIS, 3))'; 
    Y_S_2d = reshape(double(MSI), [], size(MSI, 3))';    
    
    %% AVIRIS <-> Landsat-8/9
    AVI_wavelength = [365.9298, 375.5939, 385.2625, 394.9355, 404.6128, 414.2946, 423.9807, 433.6713, 443.3662, 453.0655, 462.7692, 472.4772, 482.1897, 491.9066, 501.6279, 511.3535, 521.0835, 530.8179, 540.5568, 550.3000, 560.0476, 569.7996, 579.5560, 589.3167, 599.0819, 608.8515, 618.6254, 628.4037, 638.1864, 647.9735, 657.7650, 667.5609, 654.7923, 664.5993, 674.4012, 684.1979, 693.9893, 703.7756, 713.5566, 723.3325, 733.1030, 742.8685, 752.6287, 762.3837, 772.1334, 781.8780, 791.6174, 801.3515, 811.0805, 820.8042, 830.5227, 840.2360, 849.9441, 859.6470, 869.3447, 879.0372, 888.7244, 898.4065, 908.0833, 917.7550, 927.4214, 937.0827, 946.7387, 956.3894, 966.0351, 975.6754, 985.3106, 994.9405, 1004.5653, 1014.1848, 1023.7991, 1033.4083, 1043.0122, 1052.6109, 1062.2044, 1071.7927, 1081.3757, 1090.9536, 1100.5262, 1110.0937, 1119.6560, 1129.2130, 1138.7647, 1148.3114, 1157.8527, 1167.3890, 1176.9199, 1186.4458, 1195.9663, 1205.4816, 1214.9918, 1224.4967, 1233.9964, 1243.4909, 1252.7729, 1252.9802, 1262.4643, 1262.7457, 1272.7182, 1282.6905, 1292.6624, 1302.6342, 1312.6058, 1322.5771, 1332.5481, 1342.5189, 1352.4895, 1362.4598, 1372.4299, 1382.3997, 1392.3693, 1402.3387, 1412.3078, 1422.2767, 1432.2453, 1442.2137, 1452.1818, 1462.1499, 1472.1175, 1482.0849, 1492.0521, 1502.0190, 1511.9858, 1521.9522, 1531.9184, 1541.8845, 1551.8502, 1561.8156, 1571.7810, 1581.7459, 1591.7108, 1601.6752, 1611.6396, 1621.6036, 1631.5675, 1641.53101, 1651.4943, 1661.4575, 1671.4202, 1681.3829, 1691.3453, 1701.3073, 1711.2692, 1721.2309, 1731.1923, 1741.1534, 1751.1143, 1761.0750, 1771.0355, 1780.9957, 1790.9556, 1800.9154, 1810.8748, 1820.8341, 1830.7930, 1840.7518, 1850.7103, 1860.6685, 1870.6265, 1871.7843, 1865.9637, 1876.0252, 1886.0845, 1896.1414, 1906.1960, 1916.2484, 1926.29846 , 1936.3460699999998 , 1946.39148 , 1956.4345700000001 , 1966.47534 , 1976.51379 , 1986.55005 , 1996.5838600000002 , 2006.6153, 2016.6446, 2026.6716, 2036.6961, 2046.71863 , 2056.73877 , 2066.75635 , 2076.77173 , 2086.78491 , 2096.7956499999996 , 2106.8042 , 2116.8103 , 2126.8142, 2136.8156, 2146.8149, 2156.8120, 2166.8066, 2176.7990, 2186.7890, 2196.7766, 2206.7622, 2216.7453, 2226.7260, 2236.7045, 2246.6806, 2256.6545, 2266.6262, 2276.5954, 2286.5625, 2296.5271, 2306.4895, 2316.4494, 2326.4072, 2336.3625, 2346.3156, 2356.2666, 2366.2150, 2376.1611, 2386.1052, 2396.0466, 2405.9860, 2415.9231 , 2425.8576, 2435.7900, 2445.7199, 2455.6477, 2465.5732, 2475.4963, 2485.4172, 2495.3356]/ 1000;
    removed_idx = [1:10, 104:116, 152:170, 215:224];
    valid_idx = setdiff(1:224, removed_idx);
    AVI_wavelength = AVI_wavelength(valid_idx);
    Landsat_wavelength = [443, 482.5, 562.5, 655, 865, 1610, 2200] / 1000;
    Landsat_map_AVI = zeros(size(Landsat_wavelength));
    Landsat_band_idx_in_AVIRIS = zeros(size(Landsat_wavelength));
    for i = 1:length(Landsat_wavelength)
        [~, idx] = min(abs(AVI_wavelength - Landsat_wavelength(i)));
        Landsat_map_AVI(i) = AVI_wavelength(idx);
        Landsat_band_idx_in_AVIRIS(i) = idx;
    end
    Landsat_wavelength = Landsat_map_AVI;
    
    %% random pixels: spectral curve & calculate SAM
    GT_selected = GT_2d(:, random_indices);
    Y_S_selected = Y_S_2d(:, random_indices);
    sam_gt = zeros(num_random_pixels, 1);
    
    for i = 1:num_random_pixels
        [y, x] = ind2sub([H, W], random_indices(i));
        figure('Position', [100, 100, 1200, 600]);
        GT_i = GT_selected(:, i);
        Y_Si = Y_S_selected(:, i);
        curve_output = squeeze(model(y, x, :));
        sam_gt(i) = calculate_sam(GT_i, curve_output);
        
        seg1 = AVI_wavelength(AVI_wavelength < 1.4);
        seg2 = AVI_wavelength(AVI_wavelength > 1.4 & AVI_wavelength < 1.8);
        seg3 = AVI_wavelength(AVI_wavelength > 1.8);
        x1 = seg1; x2 = seg2; x3 = seg3; 
        x_axis = [x1, x2, x3];
    
        h_gt = plot(x_axis, GT_i, 'k-','LineWidth', 1.5, 'DisplayName', 'AVIRIS');
        hold on;
        h_output = plot(x_axis, curve_output, 'r-', 'LineWidth', 1.5, 'DisplayName', 'PAINT Result');
        
        for ii = 1:length(Landsat_wavelength)
            if ii == 1
                h_landsat = plot([Landsat_wavelength(ii), Landsat_wavelength(ii)], [0, Y_Si(ii)], 'b-', 'LineWidth', 1.5, 'DisplayName', 'Landsat-8/9 Curve');
            else
                plot([Landsat_wavelength(ii), Landsat_wavelength(ii)], [0, Y_Si(ii)], 'b-', 'LineWidth', 1.5, 'HandleVisibility', 'off');
            end
        end
        
        font_size = 36; font_name = 'Times New Roman';
        tick_positions = [x1(1), 1.5, x3(end)];
        xticks(tick_positions);
        xticklabels({'0.5', '1.5', '2.5'});
        xlabel("Wavelength (μm)")
        ylabel("Reflectance")
        title(sprintf("Pixel Value at (%d, %d)", cols(i), rows(i)))
        xlim([min(x_axis)-0.05, max(x_axis)+0.05]);
        ymax = max([GT_i(:); curve_output(:)]) + 0.05;
        ylim([0, ymax]);
        
        ax = gca;
        ax.FontSize = font_size;
        ax.FontName = font_name;
        grid on;
        legend([h_gt, h_output, h_landsat], ...
            {'AVIRIS', 'PAINT Result', 'Landsat-8/9'}, ...
            'FontName', font_name, ...
            'FontSize', 18, ...
            'Location', 'best', ...
            'Box', 'off');
    end
    
    %% img
    band_set = [23 12 5];  
    plot_comparison_images(AVIRIS, model, band_set);
    calculate_and_display_metrics(double(AVIRIS), double(model));
end
    
%% subfunction: Calculate SAM (Spectral Angle Mapper) in degrees
function sam_degrees = calculate_sam(vector1, vector2)
    dot_product = sum(vector1 .* vector2);
    norm1 = norm(vector1);
    norm2 = norm(vector2);
    cosine_value = dot_product / (norm1 * norm2);
    cosine_value = min(1, max(-1, cosine_value));
    sam_degrees = acos(cosine_value) * 180/pi;
end

%% subfunction: Plot images
function plot_comparison_images(AVIRIS, output, band_set)
    random_indices = evalin('base', 'random_indices');
    [H, W, ~] = evalin('base', 'size(AVIRIS)');
    saved_y = zeros(length(random_indices), 1);
    saved_x = zeros(length(random_indices), 1);
    for i = 1:length(random_indices)
        [saved_y(i), saved_x(i)] = ind2sub([H, W], random_indices(i));
    end
    
    % normalization
    normColor=@(R)max(min((R-mean(R(:)))/std(R(:)),2),-2)/3+0.5;
    figure('Position', [0 0 1200 800]); 

    % (a) Ground Truth
    subplot(1,3,1);
    temp_show = output(:,:,band_set);
    temp_show = normColor(temp_show);
    imshow(temp_show);
    title('Reference HSI', ...
    'FontName', 'Times New Roman', ...
    'FontSize', 14, ...
    'FontWeight', 'normal');
    
    % (b) Sample Points
    subplot(1,3,2);
    temp_show = AVIRIS(:,:,band_set);
    temp_show = normColor(temp_show);
    imshow(temp_show);
    hold on;
    
    % Initialize global variables for dragging functionality
    global dragging current_text
    dragging = false;
    current_text = [];
    
    % Store text handles for dragging functionality
    text_handles = gobjects(length(random_indices), 1);
    
    % Plot the random points
    for i = 1:length(random_indices)
        scatter(saved_x(i), saved_y(i),200,[1.0,0.0,0.0], 'o', 'filled');
        [text_x, text_y] = calculateLabelPosition(saved_x(i), saved_y(i), W, H);
        text_handles(i) = text(text_x, text_y, sprintf('(%d,%d)', saved_x(i), saved_y(i)), ...
            'Color', 'black', 'FontSize', 18,'FontName', 'Times New Roman', 'FontWeight', 'bold', ...
            'BackgroundColor', [1 1 1 0.6], 'Clipping', 'on', 'ButtonDownFcn', @startDragText);
    end
    hold off;
    set(gcf, 'WindowButtonMotionFcn', @dragText);
    set(gcf, 'WindowButtonUpFcn', @stopDragText);
    title('Reference HSI with Sampled Pixel', ...
    'FontName', 'Times New Roman', ...
    'FontSize', 14, ...
    'FontWeight', 'normal');

    % (c) Orig Low-res HSI(256*256)
    subplot(1,3,3);
    temp_show = AVIRIS(:,:,band_set);
    temp_show = normColor(temp_show);
    imshow(temp_show);
    title('PAINT-reconstructed HSI', ...
    'FontName', 'Times New Roman', ...
    'FontSize', 14, ...
    'FontWeight', 'normal');
end

%% Callback functions for dragging text labels
function startDragText(src, ~)
    global dragging current_text
    dragging = true;
    current_text = src;
    % Change cursor to indicate dragging
    set(gcf, 'Pointer', 'hand');
end

function dragText(~, ~)
    global dragging current_text
    % Add explicit checks to ensure variables are properly initialized
    if isempty(dragging)
        dragging = false;
    end
    if isempty(current_text)
        current_text = [];
    end
    
    % Use logical() to ensure dragging is a logical scalar
    if logical(dragging) && ~isempty(current_text) && ishandle(current_text)
        % Get current cursor position in axes coordinates
        ax = gca;
        cp = get(ax, 'CurrentPoint');
        
        % Get image dimensions for boundary checking
        img_width = 256;
        img_height = 256;
        
        % Update text position with boundary constraints
        new_x = max(1, min(img_width, cp(1,1)));
        new_y = max(1, min(img_height, cp(1,2)));
        
        set(current_text, 'Position', [new_x, new_y]);
    end
end

function stopDragText(~, ~)
    global dragging current_text
    % Ensure variables are properly initialized
    if isempty(dragging)
        dragging = false;
    end
    
    if logical(dragging)
        dragging = false;
        current_text = [];
        % Reset cursor
        set(gcf, 'Pointer', 'arrow');
    end
end

%% Helper function to calculate optimal label position
function [text_x, text_y] = calculateLabelPosition(point_x, point_y, img_width, img_height)
    % Parameters for label positioning
    offset = 8;         % Standard offset from point
    text_width = 55;    % Approximate width of text label
    text_height = 15;   % Approximate height of text label
    
    % Start with default position (to the right of point)
    text_x = point_x + offset;
    text_y = point_y;
    
    % EDGE CASE HANDLING - More conservative boundary checks
    % Right edge handling
    if point_x > img_width - text_width - 5
        text_x = point_x - text_width - offset; % Place to the left
    end
    
    % Bottom edge handling - stricter boundary check
    if point_y > img_height - text_height - 10
        text_y = point_y - text_height - offset; % Place above
    end
    
    % Top edge handling
    if point_y < text_height + 5
        text_y = point_y + offset; % Place below
    end
    
    % Left edge handling
    if point_x < text_width + 5
        text_x = point_x + offset; % Ensure text is to the right
    end
    
    % Corner cases
    % Bottom right corner
    if point_x > img_width - text_width - 5 && point_y > img_height - text_height - 10
        text_x = point_x - text_width - offset;
        text_y = point_y - text_height - offset;
    end
    
    % Bottom left corner
    if point_x < text_width + 5 && point_y > img_height - text_height - 10
        text_x = point_x;
        text_y = point_y - text_height - offset;
    end
end

%% subfunction: Calculate and display metrics for output and Z_fused compared to X
function calculate_and_display_metrics(ref, model)
    % Get dimensions
    [h, w, bands] = size(ref);
    L = h * w; 

    % Reshape for pixel-wise calculations
    ref_reshaped = reshape(ref, [], bands);
    output_reshaped = reshape(model, [], bands);
    
    % Reshape back to 3D for band-wise calculations
    ref_3d = reshape(ref_reshaped, h, w, bands);
    output_3d = reshape(output_reshaped, h, w, bands);
    
    % 1. PSNR calculation
    mse_output = mean((ref_reshaped - output_reshaped).^2, 1);
    max_val_output = max(output_reshaped, [], 1).^2;
    psnr_output = mean(10*log10(max_val_output./mse_output));
    
    % 2. RMSE calculation
    rmse_m_output = zeros(bands, 1);
    for i = 1:bands
        rmse_m_output(i) = sqrt(sum(sum((ref_3d(:,:,i) - output_3d(:,:,i)).^2))) / sqrt(L);
    end
    rmse_output = sqrt(sum(rmse_m_output.^2) / bands);
    
    % 3. SAM calculation
    sam_output = calculate_sam_metric(ref_reshaped, output_reshaped);
    
    % 4. SSIM calculation
    ssim_sum_output = 0;
    for i = 1:bands
        ssim_sum_output = ssim_sum_output + ssim(ref(:,:,i), model(:,:,i));
    end
    ssim_output = ssim_sum_output / bands;
    
    % Display results in a formatted table
    disp('========================================================================');
    fprintf('PSNR (dB): %.4f   SAM (deg): %.4f   RMSE: %.4f   SSIM: %.4f \n', psnr_output, sam_output, rmse_output, ssim_output);
    disp('========================================================================'); 
end

%% Helper function to calculate SAM metric for two image matrices
function sam_value = calculate_sam_metric(img1_reshaped, img2_reshaped)
    sam_sum = 0;
    for i = 1:size(img1_reshaped, 1)
        dot_product = sum(img1_reshaped(i,:) .* img2_reshaped(i,:));
        norm_img1 = norm(img1_reshaped(i,:));
        norm_img2 = norm(img2_reshaped(i,:));
        if norm_img1 > 0 && norm_img2 > 0
            cosine_value = dot_product/(norm_img1*norm_img2);
            cosine_value = min(1, max(-1, cosine_value));
            sam_sum = sam_sum + acos(cosine_value);
        end
    end
    sam_value = sam_sum / size(img1_reshaped, 1) * 180/pi;
end