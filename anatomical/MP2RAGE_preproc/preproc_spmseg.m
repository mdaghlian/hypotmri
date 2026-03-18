function preproc_spmseg(full_path_to_file, full_path_to_out)
disp(' ');
disp('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++');
disp([datestr(datetime('now')),'        Starting SPM Segmentation']);
disp('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++');
disp(' ');

%% Check if SPM Directory exists on path
if exist('spm') == 0
    disp('++++ SPM directory not found in path.');
    disp(' ');
    spm_directory = uigetdir(pwd, 'Select directory with SPM 12');
    addpath(spm_directory);
    disp(['> ', spm_directory]);
    disp('> Added to path');
else
    spm_directory = which('spm');
    spm_directory = spm_directory(1:end-6);
    disp('++++ SPM directory exists in path.');
    disp(['> ', spm_directory]);
end

%% Select Data
if exist('full_path_to_file', 'var') == 1
    disp(' ');
    disp('++++ Input File Provided.');
    disp(['> ', full_path_to_file]);
else
    [in_file_name, in_file_path] = uigetfile('*.nii;*.nii.gz', 'Select Input T1w File');
    disp(' ');
    disp('++++ Input File Selected.');
    full_path_to_file = fullfile(in_file_path, in_file_name);
    disp(['> ', full_path_to_file]);
end

%% Set output directory
[in_file_path, in_file_prefix, in_file_ext] = fileparts(full_path_to_file);
if exist('full_path_to_out', 'var') == 0
    full_path_to_out = fullfile(in_file_path, [in_file_prefix, '_spmseg']);
end
mkdir(full_path_to_out);
disp(' ');
disp('++++ Output Directory Created.');
disp(['> ', full_path_to_out]);

%% Handle .gz input
if strcmp(in_file_ext, '.gz')
    disp(' ');
    disp('++++ Unzipping Input file');
    disp(['> ', full_path_to_file]);
    gunzip(full_path_to_file, full_path_to_out);
    [~, nii_name, ~] = fileparts(full_path_to_file);
    in_file_name = nii_name;
    disp('++++ Unzipped Input file');
    full_path_to_file = fullfile(full_path_to_out, in_file_name);
    [~, in_file_prefix, ~] = fileparts(full_path_to_file);
    disp(['> ', full_path_to_file]);
else
    disp('++++ Input file is unzipped');
    in_file_name = [in_file_prefix, '.nii'];
    copyfile(full_path_to_file, fullfile(full_path_to_out, in_file_name));
    full_path_to_file = fullfile(full_path_to_out, in_file_name);
    disp(['> ', full_path_to_file]);
end

%% Get native voxel resolution
V_in      = spm_vol(full_path_to_file);
vox_sizes = sqrt(sum(V_in.mat(1:3,1:3).^2));
median_res = median(vox_sizes);
disp(' ');
disp(sprintf('++++ Native voxel resolution: %.3f x %.3f x %.3f mm', vox_sizes(1), vox_sizes(2), vox_sizes(3)));
disp(sprintf('++++ Median resolution: %.3f mm', median_res));

%% Setup SPM Segmentation Batch
disp(' ');
disp('++++ Setting up SPM Segmentation Batch');
clear matlabbatch;
tpm_path = fullfile(spm_directory, 'tpm', 'TPM.nii');

matlabbatch{1}.spm.spatial.preproc.channel.vols    = {[full_path_to_file, ',1']};
matlabbatch{1}.spm.spatial.preproc.channel.biasreg = 0.001;
matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = 30;
matlabbatch{1}.spm.spatial.preproc.channel.write   = [1 1];  % write bias field + bias-corrected

% GM — native only
matlabbatch{1}.spm.spatial.preproc.tissue(1).tpm    = {[tpm_path, ',1']};
matlabbatch{1}.spm.spatial.preproc.tissue(1).ngaus  = 2;
matlabbatch{1}.spm.spatial.preproc.tissue(1).native = [1 0];
matlabbatch{1}.spm.spatial.preproc.tissue(1).warped = [0 0];

% WM — native only
matlabbatch{1}.spm.spatial.preproc.tissue(2).tpm    = {[tpm_path, ',2']};
matlabbatch{1}.spm.spatial.preproc.tissue(2).ngaus  = 2;
matlabbatch{1}.spm.spatial.preproc.tissue(2).native = [1 0];
matlabbatch{1}.spm.spatial.preproc.tissue(2).warped = [0 0];

% CSF — native only
matlabbatch{1}.spm.spatial.preproc.tissue(3).tpm    = {[tpm_path, ',3']};
matlabbatch{1}.spm.spatial.preproc.tissue(3).ngaus  = 2;
matlabbatch{1}.spm.spatial.preproc.tissue(3).native = [1 0];
matlabbatch{1}.spm.spatial.preproc.tissue(3).warped = [0 0];

% Bone — native only (used for brain mask exclusion)
matlabbatch{1}.spm.spatial.preproc.tissue(4).tpm    = {[tpm_path, ',4']};
matlabbatch{1}.spm.spatial.preproc.tissue(4).ngaus  = 3;
matlabbatch{1}.spm.spatial.preproc.tissue(4).native = [1 0];
matlabbatch{1}.spm.spatial.preproc.tissue(4).warped = [0 0];

% Soft tissue — native only
matlabbatch{1}.spm.spatial.preproc.tissue(5).tpm    = {[tpm_path, ',5']};
matlabbatch{1}.spm.spatial.preproc.tissue(5).ngaus  = 4;
matlabbatch{1}.spm.spatial.preproc.tissue(5).native = [1 0];
matlabbatch{1}.spm.spatial.preproc.tissue(5).warped = [0 0];

% Air/background — native only
matlabbatch{1}.spm.spatial.preproc.tissue(6).tpm    = {[tpm_path, ',6']};
matlabbatch{1}.spm.spatial.preproc.tissue(6).ngaus  = 2;
matlabbatch{1}.spm.spatial.preproc.tissue(6).native = [1 0];
matlabbatch{1}.spm.spatial.preproc.tissue(6).warped = [0 0];

% Warp settings — no deformation fields written
matlabbatch{1}.spm.spatial.preproc.warp.mrf    = 1;
matlabbatch{1}.spm.spatial.preproc.warp.cleanup = 1;
matlabbatch{1}.spm.spatial.preproc.warp.reg    = [0 0.001 0.5 0.05 0.2];
matlabbatch{1}.spm.spatial.preproc.warp.affreg = 'mni';
matlabbatch{1}.spm.spatial.preproc.warp.fwhm   = 0;
matlabbatch{1}.spm.spatial.preproc.warp.samp   = 2;
matlabbatch{1}.spm.spatial.preproc.warp.write  = [0 0];
matlabbatch{1}.spm.spatial.preproc.warp.vox    = NaN;
matlabbatch{1}.spm.spatial.preproc.warp.bb     = [NaN NaN NaN; NaN NaN NaN];

%% Run SPM Segmentation
disp(' ');
disp('++++ Starting SPM Segmentation');
spm('defaults', 'FMRI');
spm_jobman('run', matlabbatch);
save(fullfile(full_path_to_out, [in_file_prefix, '_spmseg_batch.mat']), 'matlabbatch');

disp(' ');
disp('++++ SPM Segmentation complete. Reorganising outputs...');

%% Rename outputs
% Bias-corrected image (m prefix)
bc_src = fullfile(full_path_to_out, ['m', in_file_name]);
bc_dst = fullfile(full_path_to_out, [in_file_prefix, '_biascorrected.nii']);
if exist(bc_src, 'file')
    movefile(bc_src, bc_dst);
    disp(['++++ Renamed bias-corrected image: ', bc_dst]);
end

% Bias field (BiasField_ prefix)
bf_src = fullfile(full_path_to_out, ['BiasField_', in_file_name]);
bf_dst = fullfile(full_path_to_out, [in_file_prefix, '_biasfield.nii']);
if exist(bf_src, 'file')
    movefile(bf_src, bf_dst);
    disp(['++++ Renamed bias field: ', bf_dst]);
end

% seg8 mat — keep it, just rename
seg8_src = fullfile(full_path_to_out, [in_file_prefix, '_seg8.mat']);
seg8_dst = fullfile(full_path_to_out, [in_file_prefix, '_spmseg8.mat']);
if exist(seg8_src, 'file')
    movefile(seg8_src, seg8_dst);
    disp(['++++ Renamed seg8 mat: ', seg8_dst]);
end

% GM (c1)
gm_src = fullfile(full_path_to_out, ['c1', in_file_name]);
gm_dst = fullfile(full_path_to_out, [in_file_prefix, '_GM_native.nii']);
if exist(gm_src, 'file')
    movefile(gm_src, gm_dst);
    disp(['++++ Renamed GM map: ', gm_dst]);
end

% WM (c2)
wm_src = fullfile(full_path_to_out, ['c2', in_file_name]);
wm_dst = fullfile(full_path_to_out, [in_file_prefix, '_WM_native.nii']);
if exist(wm_src, 'file')
    movefile(wm_src, wm_dst);
    disp(['++++ Renamed WM map: ', wm_dst]);
end

% CSF (c3)
csf_src = fullfile(full_path_to_out, ['c3', in_file_name]);
csf_dst = fullfile(full_path_to_out, [in_file_prefix, '_CSF_native.nii']);
if exist(csf_src, 'file')
    movefile(csf_src, csf_dst);
    disp(['++++ Renamed CSF map: ', csf_dst]);
end

% Bone (c4)
bone_src = fullfile(full_path_to_out, ['c4', in_file_name]);
bone_dst = fullfile(full_path_to_out, [in_file_prefix, '_bone_native.nii']);
if exist(bone_src, 'file')
    movefile(bone_src, bone_dst);
    disp(['++++ Renamed bone map: ', bone_dst]);
end

% Soft tissue (c5)
soft_src = fullfile(full_path_to_out, ['c5', in_file_name]);
soft_dst = fullfile(full_path_to_out, [in_file_prefix, '_softtissue_native.nii']);
if exist(soft_src, 'file')
    movefile(soft_src, soft_dst);
    disp(['++++ Renamed soft tissue map: ', soft_dst]);
end

% Air/background (c6)
air_src = fullfile(full_path_to_out, ['c6', in_file_name]);
air_dst = fullfile(full_path_to_out, [in_file_prefix, '_air_native.nii']);
if exist(air_src, 'file')
    movefile(air_src, air_dst);
    disp(['++++ Renamed air/background map: ', air_dst]);
end

%% Generate brain mask from tissue probability maps
disp(' ');
disp('++++ Generating brain mask from tissue probability maps...');

gm_file  = fullfile(full_path_to_out, [in_file_prefix, '_GM_native.nii']);
wm_file  = fullfile(full_path_to_out, [in_file_prefix, '_WM_native.nii']);
csf_file = fullfile(full_path_to_out, [in_file_prefix, '_CSF_native.nii']);
bone_file = fullfile(full_path_to_out, [in_file_prefix, '_bone_native.nii']);
soft_file = fullfile(full_path_to_out, [in_file_prefix, '_softtissue_native.nii']);
air_file  = fullfile(full_path_to_out, [in_file_prefix, '_air_native.nii']);


if exist(gm_file, 'file') && exist(wm_file, 'file') && exist(csf_file, 'file')
    V_gm  = spm_vol(gm_file);
    V_wm  = spm_vol(wm_file);
    V_csf = spm_vol(csf_file);
    V_bone = spm_vol(bone_file);
    V_soft = spm_vol(soft_file);
    V_air  = spm_vol(air_file);

    gm_vol  = spm_read_vols(V_gm);
    wm_vol  = spm_read_vols(V_wm);
    csf_vol = spm_read_vols(V_csf);
    bone_vol = spm_read_vols(V_bone);
    soft_vol = spm_read_vols(V_soft);
    air_vol  = spm_read_vols(V_air);

    se = strel('sphere', 8);

    % ------------------------------------------------------------------ %
    %  Mask 1: GM + WM + CSF >= 0.1  (original)
    % ------------------------------------------------------------------ %
    combined  = gm_vol + wm_vol + csf_vol;
    mask_gmwmcsf = close_fill_lcc(combined >= 0.1, se);

    V_mask         = V_gm;
    mask_path      = fullfile(full_path_to_out, [in_file_prefix, '_GMWMCSFbrainmask.nii']);
    V_mask.fname   = mask_path;
    V_mask.dt      = [spm_type('uint8'), 0];
    V_mask.descrip = 'Brain mask: GM+WM+CSF >= 0.1, closed + LCC';
    V_mask.pinfo   = [1; 0; 0];
    spm_write_vol(V_mask, uint8(mask_gmwmcsf));
    disp(['++++ Brain mask (GM+WM+CSF) written: ', mask_path]);
    disp(sprintf('     Brain voxels: %d  (%.1f cm^3)', ...
        sum(mask_gmwmcsf(:)), ...
        sum(mask_gmwmcsf(:)) * abs(det(V_gm.mat(1:3,1:3))) / 1000));

    % ------------------------------------------------------------------ %
    %  Mask 2: strip-mask logic  1 - (CSF+bone+soft+air > 0.5)
    %          then same closing + fill + LCC
    % ------------------------------------------------------------------ %
    strip_raw        = 1 - ((csf_vol + bone_vol + soft_vol + air_vol) > 0.5);
    mask_strip       = close_fill_lcc(strip_raw > 0, se);

    V_mask2          = V_gm;
    mask_strip_path  = fullfile(full_path_to_out, [in_file_prefix, '_stripbrainmask.nii']);
    V_mask2.fname    = mask_strip_path;
    V_mask2.dt       = [spm_type('uint8'), 0];
    V_mask2.descrip  = 'Brain mask: strip logic (1-(CSF+bone+soft+air>0.5)), closed + LCC';
    V_mask2.pinfo    = [1; 0; 0];
    spm_write_vol(V_mask2, uint8(mask_strip));
    disp(['++++ Brain mask (strip-based) written: ', mask_strip_path]);
    disp(sprintf('     Brain voxels: %d  (%.1f cm^3)', ...
        sum(mask_strip(:)), ...
        sum(mask_strip(:)) * abs(det(V_gm.mat(1:3,1:3))) / 1000));

else
    warning('s02_spmseg: Could not find one or more tissue maps — brain mask not created.');
    disp('     Expected: _GM_native.nii, _WM_native.nii, _CSF_native.nii');
end
%% Fin
disp(' ');
disp('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++');
disp([datestr(datetime('now')),'        Completed SPM Segmentation']);
disp('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++');
disp(' ');
end



% ------------------------------------------------------------------ %
%  Local helper
% ------------------------------------------------------------------ %
function out_mask = close_fill_lcc(in_mask, se)
    out_mask = imclose(in_mask, se);
    for z = 1:size(out_mask, 3)
        out_mask(:,:,z) = imfill(out_mask(:,:,z), 'holes');
    end
    out_mask = imfill(out_mask, 'holes');
    CC       = bwconncomp(out_mask, 26);
    num_vox  = cellfun(@numel, CC.PixelIdxList);
    [~, idx] = max(num_vox);
    out_mask = false(size(out_mask));
    out_mask(CC.PixelIdxList{idx}) = true;
end