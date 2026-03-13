function s02_cat12seg(full_path_to_file, full_path_to_out)
disp(' ');
disp('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++');
disp([datestr(datetime('now')),'        Starting CAT12 Segmentation']);
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
    spm_directory = spm_directory(1:end - 6);
    disp('++++ SPM directory exists in path.');
    disp(['> ', spm_directory]);
end

%% Check if CAT12 Directory exists on path
cat12_directory = fullfile(spm_directory, 'toolbox', 'cat12');
if exist(cat12_directory, 'dir') == 0
    disp('++++ CAT12 directory not found in default SPM toolbox path.');
    disp(' ');
    cat12_directory = uigetdir(pwd, 'Select directory with CAT12');
    addpath(cat12_directory);
    disp(['> ', cat12_directory]);
    disp('> Added to path');
else
    addpath(cat12_directory);
    disp('++++ CAT12 directory exists.');
    disp(['> ', cat12_directory]);
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
    full_path_to_out = fullfile(in_file_path, [in_file_prefix, '_cat12seg']);
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
    % Strip the .gz to get the .nii filename
    [~, nii_name, ~] = fileparts(full_path_to_file);   % nii_name = prefix.nii
    in_file_name = nii_name;
    disp('++++ Unzipped Input file');
    full_path_to_file = fullfile(full_path_to_out, in_file_name);
    [~, in_file_prefix, ~] = fileparts(full_path_to_file);
    disp(['> ', full_path_to_file]);
else
    disp('++++ Input file is unzipped');
    in_file_name = [in_file_prefix, '.nii'];
    % Copy .nii to output directory
    copyfile(full_path_to_file, fullfile(full_path_to_out, in_file_name));
    full_path_to_file = fullfile(full_path_to_out, in_file_name);
    disp(['> ', full_path_to_file]);
end

%% MP2RAGE-specific parameters
% APP = 1070: full preprocessing pipeline with T2* contrast correction (recommended for MP2RAGE)
% NCstr = -Inf: adaptive SANLM noise correction strength (CAT12 chooses optimal level)
% biasstr = 0.5: medium bias correction strength (MP2RAGE is already largely bias-free,
%                lower values preserve more original intensity; increase if residual bias visible)
APP_term   = 1070;  % MP2RAGE: handles inverted background noise correctly
SANLM_term = -Inf;  % Adaptive noise correction strength
biasstr    = 0.5;   % Bias field correction strength

%% Setup CAT12 Segmentation Batch
disp(' ');
disp('++++ Setting up CAT12 Segmentation Batch');
clear matlabbatch;

% -----------------------------------------------------------------------
% Load CAT12 defaults then build the job from them.
% This ensures field names are always valid for the installed version.
% We only override the fields we explicitly want to change.
% -----------------------------------------------------------------------
cat_get_defaults;
job = cat_get_defaults('extopts');   % returns extopts struct

% --- Top-level ---
matlabbatch{1}.spm.tools.cat.estwrite.data     = {[full_path_to_file, ',1']};
matlabbatch{1}.spm.tools.cat.estwrite.data_wmh = {''};
matlabbatch{1}.spm.tools.cat.estwrite.nproc    = 0;
matlabbatch{1}.spm.tools.cat.estwrite.useprior = '';

% --- opts (SPM preprocessing) ---
matlabbatch{1}.spm.tools.cat.estwrite.opts.tpm    = {fullfile(spm_directory, 'tpm', 'TPM.nii')};
matlabbatch{1}.spm.tools.cat.estwrite.opts.affreg = 'mni';

% biasstr / accstr field names differ between CAT12 r1xxx and r2xxx.
% Use cat_get_defaults to write them safely.
cat_get_defaults('opts.biasstr', biasstr);
cat_get_defaults('opts.accstr',  0.5);
matlabbatch{1}.spm.tools.cat.estwrite.opts = cat_get_defaults('opts');
% Restore tpm/affreg which cat_get_defaults may have overwritten with its own paths
matlabbatch{1}.spm.tools.cat.estwrite.opts.tpm    = {fullfile(spm_directory, 'tpm', 'TPM.nii')};
matlabbatch{1}.spm.tools.cat.estwrite.opts.affreg = 'mni';

% --- extopts: start from CAT12 defaults, then override ---
extopts = cat_get_defaults('extopts');

% MP2RAGE overrides
extopts.APP    = APP_term;
extopts.NCstr  = SANLM_term;
extopts.LASstr = 0;

% Skull-stripping, cleanup, noise
if isfield(extopts, 'gcutstr');     extopts.gcutstr    = 2;   end
if isfield(extopts, 'GCUTstr');     extopts.GCUTstr    = 2;   end
if isfield(extopts, 'cleanupstr');  extopts.cleanupstr = 0.5; end
if isfield(extopts, 'BVCstr');      extopts.BVCstr     = 0.5; end
if isfield(extopts, 'LASmyostr');   extopts.LASmyostr  = 0;   end
extopts.WMHC = 0;
extopts.SLC  = 0;
extopts.mrf  = 1;

% restypes — use native resolution
if isfield(extopts, 'restypes')
    extopts.restypes = struct('native', struct());
end

% Registration — shooting template
tpm_gs = fullfile(cat12_directory, 'templates_MNI152NLin2009cAsym', 'Template_0_GS.nii');
if isfield(extopts, 'registration')
    reg = extopts.registration;
    if isfield(reg, 'regmethod')
        % newer CAT12: extopts.registration.regmethod.shooting.*
        reg.regmethod.shooting.shootingtpm = {tpm_gs};
        reg.regmethod.shooting.regstr      = 0.5;
    elseif isfield(reg, 'shooting')
        % older CAT12: extopts.registration.shooting.*
        reg.shooting.shootingtpm = {tpm_gs};
        reg.shooting.regstr      = 0.5;
    end
    reg.vox = 1.5;
    reg.bb  = 12;
    extopts.registration = reg;
elseif isfield(extopts, 'shooting')
    % flat older layout
    extopts.shooting.shootingtpm = {tpm_gs};
    extopts.shooting.regstr      = 0.5;
    extopts.vox = 1.5;
    extopts.bb  = 12;
end

% Surface options
if isfield(extopts, 'surface')
    extopts.surface.pbtres         = 0.5;
    extopts.surface.pbtmethod      = 'pbtsimple';
    extopts.surface.SRP            = 22;
    extopts.surface.vdist          = 2;
    extopts.surface.scale_cortex   = 0.7;
    extopts.surface.add_parahipp   = 0.1;
    extopts.surface.close_parahipp = 1;
else
    % flat layout
    extopts.pbtres         = 0.5;
    extopts.pbtmethod      = 'pbtsimple';
    extopts.SRP            = 22;
    extopts.vdist          = 2;
    extopts.scale_cortex   = 0.7;
    extopts.add_parahipp   = 0.1;
    extopts.close_parahipp = 1;
end

% Admin
if isfield(extopts, 'admin')
    extopts.admin.experimental = 0;
    extopts.admin.new_release  = 0;
    extopts.admin.lazy         = 0;
    extopts.admin.ignoreErrors = 1;
    extopts.admin.verb         = 2;
    extopts.admin.print        = 2;
else
    extopts.experimental = 0;
    extopts.new_release  = 0;
    extopts.lazy         = 0;
    extopts.ignoreErrors = 1;
    extopts.verb         = 2;
    extopts.print        = 2;
end

matlabbatch{1}.spm.tools.cat.estwrite.extopts = extopts;

% --- output: start from CAT12 defaults, then override ---
output = cat_get_defaults('output');

% Tissues — native only, no warped/modulated/dartel
output.GM.native  = 1;
output.WM.native  = 1;
output.CSF.native = 1;
% Zero out any warped/mod/dartel fields that exist in this version
for tissue = {'GM','WM','CSF','WMH','SL','TPMC','ct','pp','las'}
    t = tissue{1};
    if isfield(output, t)
        if isfield(output.(t), 'warped'); output.(t).warped = 0; end
        if isfield(output.(t), 'mod');    output.(t).mod    = 0; end
        if isfield(output.(t), 'dartel'); output.(t).dartel = 0; end
    end
end

% Label map (p0)
if isfield(output, 'label')
    output.label.native = 1;
    if isfield(output.label, 'warped'); output.label.warped = 0; end
    if isfield(output.label, 'dartel'); output.label.dartel = 0; end
end
if isfield(output, 'labelnative'); output.labelnative = 1; end

% Bias-corrected T1 native
if isfield(output, 'bias')
    output.bias.native = 1;
    if isfield(output.bias, 'warped'); output.bias.warped = 0; end
    if isfield(output.bias, 'dartel'); output.bias.dartel = 0; end
end

% No surfaces, no warps, no Jacobian
if isfield(output, 'surface');         output.surface         = 0; end
if isfield(output, 'surf_measures');   output.surf_measures   = 1; end
if isfield(output, 'jacobianwarped');  output.jacobianwarped  = 0; end
if isfield(output, 'warps');           output.warps           = [0 0]; end
if isfield(output, 'rmat');            output.rmat            = 0; end

% BIDS — no BIDS naming
if isfield(output, 'BIDS')
    output.BIDS.BIDSno = 1;
end

% ROI menus — no ROI
if isfield(output, 'ROImenu')
    output.ROImenu = struct('noROI', struct());
end
if isfield(output, 'sROImenu')
    output.sROImenu = struct('noROI', struct());
end

% Atlas
if isfield(output, 'atlas'); output.atlas.native = 0; end

matlabbatch{1}.spm.tools.cat.estwrite.output = output;

%% Run CAT12 Segmentation
disp(' ');
disp('++++ Starting CAT12 Segmentation');
spm('defaults', 'FMRI');
spm_jobman('run', matlabbatch);
save(fullfile(full_path_to_out, [in_file_prefix, '_cat12seg_batch.mat']), 'matlabbatch');
disp(' ');
disp('++++ CAT12 Segmentation complete. Reorganising outputs...');

%% Move CAT12 outputs from sub-folders into output directory
% CAT12 writes outputs into subdirectories (mri/, surf/, report/, label/)
% relative to the input file location. Here we copy them up to full_path_to_out.

subdirs = {'mri', 'surf', 'report', 'label'};
for i = 1:numel(subdirs)
    src_dir = fullfile(full_path_to_out, subdirs{i});
    if exist(src_dir, 'dir')
        movefile(fullfile(src_dir, '*'), full_path_to_out);
        rmdir(src_dir);
        disp(['++++ Moved contents of ', subdirs{i}, '/ to output directory.']);
    end
end

%% Rename key output files for clarity
% Bias-corrected native T1 (m<file>) — only written if output_bias = 1
bc_native_src = fullfile(full_path_to_out, ['m', in_file_name]);
bc_native_dst = fullfile(full_path_to_out, [in_file_prefix, '_biascorrected.nii']);
if exist(bc_native_src, 'file')
    movefile(bc_native_src, bc_native_dst);
    disp(['++++ Renamed bias-corrected native T1: ', bc_native_dst]);
end

% Native GM probability map (p1)
gm_src = fullfile(full_path_to_out, ['p1', in_file_name]);
gm_dst = fullfile(full_path_to_out, [in_file_prefix, '_GM_native.nii']);
if exist(gm_src, 'file')
    movefile(gm_src, gm_dst);
    disp(['++++ Renamed GM native map: ', gm_dst]);
end

% Native WM probability map (p2)
wm_src = fullfile(full_path_to_out, ['p2', in_file_name]);
wm_dst = fullfile(full_path_to_out, [in_file_prefix, '_WM_native.nii']);
if exist(wm_src, 'file')
    movefile(wm_src, wm_dst);
    disp(['++++ Renamed WM native map: ', wm_dst]);
end

% Native CSF probability map (p3)
csf_src = fullfile(full_path_to_out, ['p3', in_file_name]);
csf_dst = fullfile(full_path_to_out, [in_file_prefix, '_CSF_native.nii']);
if exist(csf_src, 'file')
    movefile(csf_src, csf_dst);
    disp(['++++ Renamed CSF native map: ', csf_dst]);
end

% Label map (p0 — tissue class: 1=CSF, 2=GM, 3=WM)
label_src = fullfile(full_path_to_out, ['p0', in_file_name]);
label_dst = fullfile(full_path_to_out, [in_file_prefix, '_label_native.nii']);
if exist(label_src, 'file')
    movefile(label_src, label_dst);
    disp(['++++ Renamed label map: ', label_dst]);
end

%% Generate brain mask from CAT12 tissue probability maps
% Strategy: sum GM + WM + CSF probability maps, threshold at 0.5,
% morphologically close to fill sulcal gaps, then keep only the
% largest connected component to remove stray voxels.
disp(' ');
disp('++++ Generating brain mask from tissue probability maps...');

% Prefer the already-renamed files; fall back to p1/p2/p3 names if
% renaming above was skipped (e.g. files already existed).
gm_file  = fullfile(full_path_to_out, [in_file_prefix, '_GM_native.nii']);
wm_file  = fullfile(full_path_to_out, [in_file_prefix, '_WM_native.nii']);
csf_file = fullfile(full_path_to_out, [in_file_prefix, '_CSF_native.nii']);

if ~exist(gm_file,  'file'); gm_file  = fullfile(full_path_to_out, ['p1', in_file_name]); end
if ~exist(wm_file,  'file'); wm_file  = fullfile(full_path_to_out, ['p2', in_file_name]); end
if ~exist(csf_file, 'file'); csf_file = fullfile(full_path_to_out, ['p3', in_file_name]); end

if exist(gm_file, 'file') && exist(wm_file, 'file') && exist(csf_file, 'file')

    % Load volumes via SPM
    V_gm  = spm_vol(gm_file);
    V_wm  = spm_vol(wm_file);
    V_csf = spm_vol(csf_file);

    gm_vol  = spm_read_vols(V_gm);
    wm_vol  = spm_read_vols(V_wm);
    csf_vol = spm_read_vols(V_csf);

    % Sum tissue probabilities and threshold.
    % 0.1 is intentionally permissive — at 7T the summed probability near
    % the cortical surface / sulcal crowns can be genuinely low due to
    % partial voluming with dura/CSF. We recover the correct boundary with
    % morphological closing + LCC rather than relying on a tight threshold.
    combined = gm_vol + wm_vol + csf_vol;
    mask     = combined >= 0.1;

    % Morphological closing to bridge sulcal gaps.
    % Radius 8 at 0.5 mm iso spans ~4 mm — enough to close sulcal CSF
    % without pulling in scalp. Increase further only if pial surface
    % is still being clipped after LCC.
    se   = strel('sphere', 8);
    mask = imclose(mask, se);

    % Fill any remaining interior holes slice-by-slice (axial),
    % then 3-D to catch coronal/sagittal residuals.
    for z = 1:size(mask, 3)
        mask(:,:,z) = imfill(mask(:,:,z), 'holes');
    end
    mask = imfill(mask, 'holes');

    % Keep only the largest connected component — removes stray scalp/dura islands
    CC        = bwconncomp(mask, 26);   % 26-connectivity for 3-D
    num_vox   = cellfun(@numel, CC.PixelIdxList);
    [~, idx]  = max(num_vox);
    mask      = false(size(mask));
    mask(CC.PixelIdxList{idx}) = true;

    % Write output using GM header as template (preserves affine/vox size)
    V_mask         = V_gm;
    mask_path      = fullfile(full_path_to_out, [in_file_prefix, '_brainmask.nii']);
    V_mask.fname   = mask_path;
    V_mask.dt      = [spm_type('uint8'), 0];   % binary → uint8 is sufficient
    V_mask.descrip = 'Brain mask: GM+WM+CSF >= 0.1, closed + LCC';
    V_mask.pinfo   = [1; 0; 0];               % slope=1, intercept=0
    spm_write_vol(V_mask, uint8(mask));

    disp(['++++ Brain mask written: ', mask_path]);
    disp(sprintf('     Brain voxels: %d  (%.1f cm^3)', ...
        sum(mask(:)), ...
        sum(mask(:)) * abs(det(V_gm.mat(1:3,1:3))) / 1000));

else
    warning('s02_cat12seg: Could not find one or more tissue maps — brain mask not created.');
    disp('     Expected: _GM_native.nii, _WM_native.nii, _CSF_native.nii');
end

%% Fin
disp(' ');
disp('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++');
disp([datestr(datetime('now')),'        Completed CAT12 Segmentation']);
disp('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++');
disp(' ');