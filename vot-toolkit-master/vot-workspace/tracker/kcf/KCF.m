function KCF


fprintf('begin!\n')
% *************************************************************
% VOT: Always call exit command at the end to terminate Matlab!
% *************************************************************
cleanup = onCleanup(@() exit() );

% *************************************************************
% VOT: Set random seed to a different value every time.
% *************************************************************
RandStream.setGlobalStream(RandStream('mt19937ar', 'Seed', sum(clock)));

% **********************************
% VOT: Get initialization data
% **********************************
[handle, image, region] = vot('rectangle');

    % If the provided region is a polygon ...
    if numel(region) > 4
%         x1 = round(min(region(1:2:end)));
%         x2 = round(max(region(1:2:end)));
%         y1 = round(min(region(2:2:end)));
%         y2 = round(max(region(2:2:end)));
%         region = round([x1, y1, x2 - x1, y2 - y1]);
        
        [cx, cy, w, h] = get_axis_aligned_BB(region);
        target_sz = [h w];
        pos = [cy cx]; % centre of the bounding box
        
    else
        region = round([round(region(1)), round(region(2)), ... 
        round(region(1) + region(3)) - round(region(1)), ...
        round(region(2) + region(4)) - round(region(2))]);
        target_sz=[region(4),region(3)];
        pos=[region(2),region(1)]+target_sz/2;
        
%        
    end;
%% Initialization
frame=1;
im = imread(image);
		if size(im,3) > 1,
			im = rgb2gray(im);
		end

	kernel_type = 'gaussian';
    feature_type = 'hog'; 
	show_visualization = 0;
    kernel.type = kernel_type;
	
	features.gray = false;
	features.hog = false;
    padding = 1.5;  %extra area surrounding the target
	lambda = 1e-4;  %regularization
	output_sigma_factor = 0.1;  %spatial bandwidth (proportional to target)
	
	switch feature_type
        case 'gray',
            interp_factor = 0.075;  %linear interpolation factor for adaptation

            kernel.sigma = 0.2;  %gaussian kernel bandwidth

            kernel.poly_a = 1;  %polynomial kernel additive term
            kernel.poly_b = 7;  %polynomial kernel exponent

            features.gray = true;
            cell_size = 1;

        case 'hog',
            interp_factor = 0.02;

            kernel.sigma = 0.5;

            kernel.poly_a = 1;
            kernel.poly_b = 9;

            features.hog = true;
            features.hog_orientations = 9;
            cell_size = 4;

        otherwise
            error('Unknown feature.')
    end
    assert(any(strcmp(kernel_type, {'linear', 'polynomial', 'gaussian'})), 'Unknown kernel.')
%       img_files=seq.s_frames;
%       target_sz =[seq.init_rect(1,4), seq.init_rect(1,3)];
%       target_sz =[region(4), region(3)];
%       pos=[seq.init_rect(1,2), seq.init_rect(1,1)] + floor(target_sz/2);
%    	  results= tracker(img_files, pos, target_sz, ...
%                padding, kernel, lambda, output_sigma_factor, interp_factor, ...
% 	           cell_size, features, show_visualization);

	resize_image = (sqrt(prod(target_sz)) >= 100);  %diagonal size >= threshold
    if resize_image,
			im = imresize(im, 0.5);
	end
    
	if resize_image,
		pos = floor(pos / 2);
		target_sz = floor(target_sz / 2);
	end


	%window size, taking padding into account
	window_sz = floor(target_sz * (1 + padding));
    
    output_sigma = sqrt(prod(target_sz)) * output_sigma_factor / cell_size;
	yf = fft2(gaussian_shaped_labels(output_sigma, floor(window_sz / cell_size)));

	%store pre-computed cosine window
	cos_window = hann(size(yf,1)) * hann(size(yf,2))';	
	
	
% 	if show_visualization,  %create video interface
% % 		update_visualization = show_video(img_files, video_path, resize_image);
%         update_visualization = show_video(img_files, resize_image);
%     end

    %note: variables ending with 'f' are in the Fourier domain.

% 	time = 0;  %to calculate FPS
% 	rect_position = zeros(numel(img_files), 4);  %to calculate precision

%%


% Initialize the tracker

while true
    
   if frame~=1
    % **********************************
    % VOT: Get next frame
    % **********************************
    
   
    [handle, image] = handle.frame(handle);

    if isempty(image)
        break;
    end;
    
	% Perform a tracking step, obtain new region

    %obtain a subwindow for detection at the position from last
	%frame, and convert to Fourier domain (its size is unchanged)
    
        im=imread(image);
        if size(im,3) > 1,
			im = rgb2gray(im);
		end
		if resize_image,
			im = imresize(im, 0.5);
		end
			patch = get_subwindow(im, pos, window_sz);
			zf = fft2(get_features(patch, features, cell_size, cos_window));
			
			%calculate response of the classifier at all shifts
			switch kernel.type
			case 'gaussian',
				kzf = gaussian_correlation(zf, model_xf, kernel.sigma);
			case 'polynomial',
				kzf = polynomial_correlation(zf, model_xf, kernel.poly_a, kernel.poly_b);
			case 'linear',
				kzf = linear_correlation(zf, model_xf);
			end
			response = real(ifft2(model_alphaf .* kzf));  %equation for fast detection

			%target location is at the maximum response. we must take into
			%account the fact that, if the target doesn't move, the peak
			%will appear at the top-left corner, not at the center (this is
			%discussed in the paper). the responses wrap around cyclically.
			[vert_delta, horiz_delta] = find(response == max(response(:)), 1);
			if vert_delta > size(zf,1) / 2,  %wrap around to negative half-space of vertical axis
				vert_delta = vert_delta - size(zf,1);
			end
			if horiz_delta > size(zf,2) / 2,  %same for horizontal axis
				horiz_delta = horiz_delta - size(zf,2);
			end
			pos = pos + cell_size * [vert_delta - 1, horiz_delta - 1];
    end
           %obtain a subwindow for training at newly estimated target position
		patch = get_subwindow(im, pos, window_sz);
		xf = fft2(get_features(patch, features, cell_size, cos_window));

		%Kernel Ridge Regression, calculate alphas (in Fourier domain)
		switch kernel.type
		case 'gaussian',
			kf = gaussian_correlation(xf, xf, kernel.sigma);
		case 'polynomial',
			kf = polynomial_correlation(xf, xf, kernel.poly_a, kernel.poly_b);
		case 'linear',
			kf = linear_correlation(xf, xf);
		end
		alphaf = yf ./ (kf + lambda);   %equation for fast training

		if frame == 1,  %first frame, train with a single image
            frame=0;
			model_alphaf = alphaf;
			model_xf = xf;
		else
			%subsequent frames, interpolate model
			model_alphaf = (1 - interp_factor) * model_alphaf + interp_factor * alphaf;
			model_xf = (1 - interp_factor) * model_xf + interp_factor * xf;
        end
            
        region=[pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
    if resize_image,
		region(:,1:2) = region(:,1:2) * 2;
    end
    % **********************************
    % VOT: Report position for frame
    % **********************************
    handle = handle.report(handle, region);
    
end;

% **********************************
% VOT: Output the results
% **********************************
handle.quit(handle);

end

