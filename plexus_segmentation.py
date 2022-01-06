import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio
import numpy as np
import time
import os
import argparse
import sys
import tqdm
import nibabel as nib

# UNet code from https://github.com/marius-sm/nd_unet
def get_norm(name, num_channels, dim=None):
    if name == 'bn':
        assert dim is not None, 'Please specify the dim argument (1, 2 or 3D)'
        if dim == 1:
            norm = nn.BatchNorm1d(num_channels)
        if dim == 2:
            norm = nn.BatchNorm2d(num_channels)
        if dim == 3:
            norm = nn.BatchNorm3d(num_channels)
        return norm
    elif 'gn' in name:
        num_groups = name[2:]
        if num_groups == '': num_groups = 8
        num_groups = int(num_groups)
        return nn.GroupNorm(num_groups, num_channels)
    elif name == 'in':
        return nn.GroupNorm(num_channels, num_channels)
    elif name == 'ln':
        return nn.GroupNorm(1, num_channels)
    else:
        raise ValueError(f"Normalization '{name}' not recognized. Possible values are None (no normalization), 'bn' (batch norm), 'gnx' (group norm where x is optionally the number of groups), 'in' (instance norm), 'ln' (layer norm)")

def get_non_lin(name):
    if name == 'relu':
        return nn.ReLU()
    if name == 'leaky_relu':
        return nn.LeakyReLU()
    if name == 'gelu':
        return nn.GELU()
    if name == 'elu':
        return nn.ELU()
    else:
        raise ValueError(f"Activation {name} not recognized. Possible values are 'relu', 'leaky_relu', 'gelu', 'elu'")

def get_conv(dim, *args, **kwargs):
    if dim == 1:
        return nn.Conv1d(*args, **kwargs)
    if dim == 2:
        return nn.Conv2d(*args, **kwargs)
    if dim == 3:
        return nn.Conv3d(*args, **kwargs)

def get_conv_block(dim, in_channels, out_channels, norm, non_lin, kernel_size=3):
    padding = kernel_size//2
    layers = [get_conv(dim, in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding)]
    if norm is not None:
        layers.append(get_norm(norm, num_channels=out_channels, dim=dim))
    if non_lin is not None:
        layers.append(get_non_lin(non_lin))  
    return nn.Sequential(*layers)

class UNetEncoder(nn.Module):
    def __init__(self, dim, in_channels, num_stages, initial_num_channels, norm=None, non_lin='relu', kernel_size=3, pooling='max'):
        super().__init__()

        assert pooling in ['avg', 'max'], f"pooling can be 'avg' or 'max'"

        if dim == 1:
            if pooling == 'avg':
                self.pooling = nn.AvgPool1d(2, 2)
            else:
                self.pooling = nn.MaxPool1d(2, 2)
        if dim == 2:
            if pooling == 'avg':
                self.pooling = nn.AvgPool2d(2, 2)
            else:
                self.pooling = nn.MaxPool2d(2, 2)
        if dim == 3:
            if pooling == 'avg':
                self.pooling = nn.AvgPool3d(2, 2)
            else:
                self.pooling = nn.MaxPool3d(2, 2)

        self.module_list = nn.ModuleList()
        
        for i in range(num_stages):
            block_1_in_channels = in_channels if i == 0 else (2**i)*initial_num_channels
            block_1_out_channels = (2**i)*initial_num_channels
            block_2_in_channels = block_1_out_channels
            block_2_out_channels = (2**(i+1))*initial_num_channels
            m = nn.Sequential(
                get_conv_block(dim=dim, in_channels=block_1_in_channels, out_channels=block_1_out_channels, kernel_size=kernel_size, norm=norm, non_lin=non_lin),
                get_conv_block(dim=dim, in_channels=block_2_in_channels, out_channels=block_2_out_channels, kernel_size=kernel_size, norm=norm, non_lin=non_lin)
            )
            self.module_list.append(m)     
            
    def forward(self, x):
        
        acts = []
        for m in self.module_list[:-1]:
            x = m(x)
            acts.append(x)
            x = self.pooling(x)
        x = self.module_list[-1](x)

        return x, acts

class UNetDecoder(nn.Module):
    def __init__(self, dim, out_channels, num_stages, initial_num_channels, norm=None, non_lin='relu', kernel_size=3):
        super().__init__()
        
        self.module_list = nn.ModuleList()
        
        for i in range(num_stages-1):
            block_in_channels = (2**(i+1) + (2**(i+2)))*initial_num_channels
            block_out_channels = (2**(i+1))*initial_num_channels
            m = nn.Sequential(
                get_conv_block(dim=dim, in_channels=block_in_channels, out_channels=block_out_channels, kernel_size=kernel_size, norm=norm, non_lin=non_lin),
                get_conv_block(dim=dim, in_channels=block_out_channels, out_channels=block_out_channels, kernel_size=kernel_size, norm=norm, non_lin=non_lin)
            )
            self.module_list.append(m)

        self.final_conv = get_conv(dim, 2*initial_num_channels, out_channels, 1, padding=0)
            
    def forward(self, x, acts):
        
        interpolation = 'linear'
        if x.dim() == 4:
            interpolation = 'bilinear'
        if x.dim() == 5:
            interpolation = 'trilinear'

        for y, m in zip(reversed(acts), reversed(self.module_list)):
            x = F.interpolate(x, y.shape[2:], mode=interpolation, align_corners=True)
            x = m(torch.cat([y, x], 1))
            
        x = self.final_conv(x)
            
        return x
        
class UNet(nn.Module):
    def __init__(self, dim, in_channels, out_channels, num_stages, initial_num_channels, norm=None, non_lin='relu', kernel_size=3, pooling='max'):
        super().__init__()
        
        self.encoder = UNetEncoder(dim, in_channels, num_stages, initial_num_channels, norm=norm, non_lin=non_lin, kernel_size=kernel_size, pooling=pooling)
        self.decoder = UNetDecoder(dim, out_channels, num_stages, initial_num_channels, norm=norm, non_lin=non_lin, kernel_size=kernel_size)
            
    def forward(self, x):
        
        x, acts = self.encoder(x)
        x = self.decoder(x, acts)
            
        return x

def UNet1d(*args, **kwargs):
    return UNet(1, *args, **kwargs)

def UNet2d(*args, **kwargs):
    return UNet(2, *args, **kwargs)

def UNet3d(*args, **kwargs):
    return UNet(3, *args, **kwargs)

@torch.no_grad()
def inference(x_highres, config, stage_1, stage_2, soft_overlap=True):

    # x_highres should have shape (c, D, H, W), i.e. no batch dimension

    low_res_shape = config['low_res_shape']

    x_lowres = F.interpolate(x_highres[None, ...].to(device), low_res_shape, align_corners=True, mode='trilinear')[0]

    c, d, h, w = x_lowres.shape
    c, D, H, W = x_highres.shape

    print('Running stage 1...')
    t0 = time.time()
    stage_1_output = stage_1(x_lowres[None, ...])[0]
    print(f'Done in {(time.time()-t0):.4f} seconds.')
    
    if config['num_communicating_channels'] > 0:
        stage_1_com_highres = F.interpolate(stage_1_output[None, 1:, ...], (D, H, W), align_corners=True, mode='trilinear')[0]

    stage_1_logits = stage_1_output[0, :, :, :] # has shape (d, h, w)
    stage_1_seg = stage_1_logits.sigmoid() # has shape (d, h, w)

    positive_threshold = config['positive_threshold']
    above_threshold = (stage_1_seg >= positive_threshold).int() # has shape (d, h, w)
    above_threshold_ind = torch.nonzero(above_threshold) # has shape (N, 3)
    num_above_threshold = above_threshold_ind.shape[0]

    if num_above_threshold == 0:
        print('WARNING: no voxels above threshold. Second stage will not be used.')
        stage_1_seg_highres = F.interpolate(stage_1_seg[None, None, ...], (D, H, W), align_corners=True, mode='trilinear')[0]
        return stage_1_seg_highres

    elif num_above_threshold > config['inference_max_patches']:
        print(f'WARNING: too many voxels above threshold ({num_above_threshold}). Only {config["inference_max_patches"]} will be used.')
    else:
        print(f'{num_above_threshold} voxels above threshold.')
        
    print('Running stage 2...')
    t0 = time.time()
    
    patch_size = config['patch_size']

    padding = patch_size//2
    x_highres_padded = F.pad(x_highres.to(device), (padding, padding, padding, padding, padding, padding)) # has shape (c, D', H', W')
    if config['num_communicating_channels'] > 0:
        stage_1_com_highres_padded = F.pad(stage_1_com_highres, (padding, padding, padding, padding, padding, padding)) # has shape (n_com, D', H', W')

    patches_x = []
    selections = []
    patches_com = []
    for j in range(min(num_above_threshold, config['inference_max_patches'])):
        loc_lowres = above_threshold_ind[j].float()
        loc_highres = loc_lowres/torch.tensor([d, h, w], device=device) * torch.tensor([D, H, W], device=device)
        bound_min = (loc_highres - patch_size/2.).round().int() + padding
        bound_max = bound_min+patch_size

        selection = (slice(None, None, None),)
        selection = selection + tuple([slice(bound_min[i].item(), bound_max[i].item(), None) for i in range(3)])
        patches_x.append(x_highres_padded[selection])
        selections.append(selection)
        if config['num_communicating_channels'] > 0:
            patches_com.append(stage_1_com_highres_padded[selection])

    patches_x = torch.stack(patches_x, 0)
    if config['num_communicating_channels'] > 0:
        patches_com = torch.stack(patches_com, 0)
        patches_x = torch.cat([patches_x, patches_com], 1)

    loader = torch.utils.data.DataLoader(patches_x, batch_size=4, shuffle=False, drop_last=False, num_workers=0)
    predictions = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm.tqdm(loader, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')):
            predictions.append(stage_2(batch))
    predictions = torch.cat(predictions, 0).sigmoid()

    if soft_overlap:
        mask_1d = torch.linspace(0, 2*np.pi, patch_size, device=predictions.device)
        mask_1d = 1 - mask_1d.cos()
        a, b, c = torch.meshgrid(mask_1d, mask_1d, mask_1d)
        mask = a*b*c
    else:
        mask = torch.ones(patch_size, patch_size, patch_size, device=predictions.device)

    y_pred = torch.zeros(*x_highres_padded.shape, device=predictions.device)
    y_norm = torch.zeros(*x_highres_padded.shape, device=predictions.device)
    for s, patch in zip(selections, predictions):
        y_pred[s] += patch * mask
        y_norm[s] += mask
    output = y_pred/y_norm
    output[torch.isnan(output)] = 0
    output = output[:, padding:-padding, padding:-padding, padding:-padding]
    
    print(f'Done in {(time.time()-t0):.4f} seconds.')
    
    return output.cpu()

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input', help='(Required) Path to the input T1 weighted image', required=True)
    parser.add_argument('-o','--output', help='(Optional) Path to the output segmentation mask', required=False)
    args = parser.parse_args()
    
    t00 = time.time()
    
    if not os.path.isfile('plexus_model.pth'):
        print(f'ERROR: "plexus_model.pth" not found.')
        sys.exit()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device}.')
    
    checkpoint = torch.load('plexus_model.pth', map_location=device)
    config = checkpoint['config']
    
    if (not os.path.isfile(args.input)) and not (not os.path.isdir(args.input)):
        print(f'ERROR: "{args.input}" does not exist.')
        sys.exit()
    
    output_file = None
        
    if args.output:
        if os.path.isfile(args.output):
            print(f'"{args.input}" already exists. Hit enter to overwrite, else type "no" and hit enter.')
            choice = input().lower()
            if choice == 'no':
                sys.exit()
            output_file = args.output
        elif os.path.isdir(args.output):
            output_file = os.path.join(args.output, 'plexus_mask_' + os.path.basename(args.input))
        else: output_file = args.output
    else:
        output_file = os.path.join(os.path.dirname(args.input), 'plexus_mask_' + os.path.basename(args.input))
    
    if output_file[-3:] != '.gz':
        output_file = output_file + '.gz'
    print(f'Output segmentation will be saved to "{output_file}".')
    
    print(f'Opening {args.input}...')
    t0 = time.time()
    t1 = torchio.ScalarImage(args.input)
    print(f'Done in {(time.time()-t0):.4f} seconds.')
    
    print(f'Preprocessing...')
    t0 = time.time()
    t1_preprocessed = torchio.transforms.ToCanonical()(t1)
    t1_preprocessed = torchio.transforms.RescaleIntensity(out_min_max=(-1, 1), percentiles=(0.5, 99.5))(t1_preprocessed)
    t1_preprocessed = torchio.transforms.Resample(target=(1, 1, 1), image_interpolation='linear')(t1_preprocessed)
    shape_after_resampling = t1_preprocessed.spatial_shape
    target_shape = (176, 240, 256)
    padding = []
    cropping = []
    for i in range(3):
        diff = target_shape[i] - shape_after_resampling[i]
        if diff > 0:
            padding = padding + [diff//2, diff-diff//2]
        else:
            padding = padding + [0, 0]
        if diff < 0:
            cropping = cropping + [abs(diff)//2, abs(diff)-abs(diff)//2]
        else:
            cropping = cropping + [0, 0]
    padding = tuple(padding)
    cropping = tuple(cropping)
    t1_preprocessed = torchio.transforms.Pad(padding, padding_mode=-1)(t1_preprocessed)
    t1_preprocessed = torchio.transforms.Crop(cropping)(t1_preprocessed)
    print(f'Done in {(time.time()-t0):.4f} seconds.')
        
    stage_1 = UNet3d(
        in_channels=1,
        out_channels=1 + config['num_communicating_channels'],
        num_stages=config['unet_stages'],
        initial_num_channels=config['unet_init_filters'],
        norm=config['unet_norm'],
        non_lin=config['unet_non_lin'],
        kernel_size=3,
        pooling='avg'
    ).to(device)
    stage_2 = UNet3d(
        in_channels=1 + config['num_communicating_channels'],
        out_channels=1,
        num_stages=config['unet_stages'],
        initial_num_channels=config['unet_init_filters'],
        norm=config['unet_norm'],
        non_lin=config['unet_non_lin'],
        kernel_size=3,
        pooling='avg'
    ).to(device)
    
    stage_1.load_state_dict(checkpoint['stage_1_state_dict'])
    stage_2.load_state_dict(checkpoint['stage_2_state_dict'])
    
    plexus_mask = inference(t1_preprocessed.tensor.float(), config, stage_1, stage_2, soft_overlap=True)
    
    print(f'Postprocessing...')
    t0 = time.time()
    plexus_mask = torchio.ScalarImage(
        tensor=plexus_mask,
        affine=t1_preprocessed.affine
    )
    plexus_mask = torchio.transforms.Pad(cropping, padding_mode=0)(plexus_mask)
    plexus_mask = torchio.transforms.Crop(padding)(plexus_mask)
    plexus_mask = torchio.transforms.Resample(target=t1, image_interpolation='linear')(plexus_mask)
    #plexus_mask = torchio.transforms.Resize(target_shape=t1.spatial_shape, image_interpolation='linear')(plexus_mask)

    # See the source code of torchio.transforms.ToCanonical
    initial_orientation = nib.orientations.io_orientation(t1.affine)
    array = plexus_mask.tensor.permute(1, 2, 3, 0)[:, :, :, None, :].numpy()
    nii = nib.Nifti1Image(array, plexus_mask.affine)
    reoriented = nii.as_reoriented(initial_orientation)
    array = torch.from_numpy(np.asanyarray(reoriented.dataobj).copy())[:, :, :, 0, :].permute(3, 0, 1, 2)
    plexus_mask.set_data(array)
    plexus_mask.affine = reoriented.affine
    print(f'Done in {(time.time()-t0):.4f} seconds.')
    
    plexus_mask.save(output_file)
    print(f'Plexus choroid segmentation was saved to "{output_file}".')
    print(f'Total execution time: {(time.time()-t00):.4f} seconds.')
    