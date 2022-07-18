import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-colorblind')
from hmrGC.image3d import trim_zeros
from hmrGC.dixon_imaging import MultiEcho
from matplotlib.patches import Rectangle
import copy

def plot_VARPROvoxel(signal, params, voxel, plot_components=None,
                     filename='', fig_name=0, patch=False, limit_ylim=False):
    refsignal = signal.copy()
    signal = signal[voxel[0], voxel[1], voxel[2], :][np.newaxis, np.newaxis,
                                                     np.newaxis, :]
    mask = np.ones((signal.shape[0], signal.shape[1], signal.shape[2]),
                   dtype=np.bool_)
    gandalf = MultiEcho(signal, mask, params)
    gandalf.range_fm = gandalf.range_fm / 2
    residual = gandalf.get_residual_fm()[0, 0, 0, :]
    psis = np.linspace(gandalf.range_fm[0], gandalf.range_fm[1],
                       int(np.ceil(np.ceil(np.diff(gandalf.range_fm))
                                   / gandalf.sampling_stepsize_fm)+1))
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(psis, residual, label='residual', color='black')
    if limit_ylim:
        ax1.set_ylim((0, 40000))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.rcParams.update({'font.size': 17})

    if plot_components:
        ax2 = ax1.twinx()
        signal = np.repeat(refsignal[voxel[0], voxel[1], voxel[2], :][np.newaxis, :],
                            len(psis), axis=0)[:, np.newaxis, np.newaxis]

        mask = np.ones((signal.shape[0], signal.shape[1], signal.shape[2]),
                       dtype=np.bool_)
        gandalf = MultiEcho(signal, np.ones_like(mask), params)
        gandalf.fieldmap = psis[:, np.newaxis, np.newaxis]
        gandalf.set_images()
        images = gandalf.images
        max_val = 0
        for key in images.keys():
            if not key == 'fatFraction_percent':
                ax2.plot(psis, np.abs(images[key][:, 0, 0]), label=key)
                max_val = np.max([np.max(np.abs(images[key])), max_val])

        #ax2.set_ylim(0, 3*max_val)
        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc=0)
        ax2.set_ylim((0, 400))
        ax2.set_ylabel('signal of chemical species (a.u.)')

    if patch:
        ax1.set_xlim((-200,200))
        helper_mask = np.ones_like(psis).astype(np.bool_)
        helper_mask[psis>200] = 0
        helper_mask[psis<-200] = 0
        min = np.min(residual[helper_mask])
        max = np.max(residual[helper_mask])
        ax1.set_ylim((min, max))
        ax1.vlines(x=0, ymin=min, ymax=max, color='black', linestyle='--')
        ax1.set_yticklabels([])
        ax1.set_xticklabels([])
        filename = f'{filename}_patch'
    else:
        ax1.set_xlabel(r'field-map $f_{\rm B}$ (Hz)')
        ax1.set_ylabel(r'$C(f_{\rm B})$ (a.u.)')

    plt.savefig(f'./figures/{fig_name}/{filename}.eps', bbox_inches='tight')
    plt.show()


def plot_NSA_template(fatimage, siliconeimage, total_species):
    silicone_phasewrap = 1.598 # echo spacing
    fat_phasewrap = 2.303 # echo spacing
    plt.rcParams.update({'font.size': 18})

    nx, ny = total_species.shape[0:2]
    fig, axs = plt.subplots(nx, ny, figsize=(10, 10))
    for i in range(nx):
        for j in range(ny):
            voxel = [i, j, 0]
            axs[i, j].set_title('FF: {}%, SF: {}%'.format(np.int(np.round(fatimage[i, j, 0]/total_species[i, j, 0]*100)),
                                                          np.int(np.round(siliconeimage[i, j, 0]/total_species[i, j, 0]*100))))
            if i == 0 and j == 0:
                axs[i, j].set_title('only water')
            if i == 1 and j == 0:
                axs[i, j].set_title('only silicone')

            axs[i, j].set_ylim((0,4.2))
            axs[i, j].vlines(silicone_phasewrap, ymin=0, ymax=4.2,
                             label='phase wrap, silicone', linewidth=1, color='black')
            axs[i, j].vlines(fat_phasewrap, ymin=0, ymax=4.2, linewidth=1,
                             label='phase wrap, main fat peak', linestyle='--', color='black')
            axs[i, j].set_xlabel('$\Delta$TE (ms)')
            axs[i, j].set_ylabel(r'NSA')
    fig.subplots_adjust()
    fig.tight_layout()
    return fig, axs


def plot_images(arr, cmap, planes, voxelSize_mm, position_3d, limits, filename='',
                fig_name=0, plot_cmap=True, patch=None):
    val_patch = []
    for plane in planes:
        fig = plt.figure(frameon=False)
        fig.set_size_inches(10,10)
        ax1 = plt.Axes(fig, [0., 0., 1., 1.])
        fig.add_axes(ax1)
        
        values = copy.deepcopy(arr)
        
        voxelSize_mm = voxelSize_mm
        cor_aspect = voxelSize_mm[2]/voxelSize_mm[0]
        sag_aspect = voxelSize_mm[2]/voxelSize_mm[1]
        trans_aspect = voxelSize_mm[1]/voxelSize_mm[0]
        if plane == 'coronal':
            values = np.transpose(values, [0, 2, 1])
            values = np.flip(values, axis=[1, 2])
            aspect = cor_aspect
            position = arr.shape[0]-position_3d[2]
        elif plane == 'sagittal':
            values = np.transpose(values, [1, 2, 0])
            values = np.flip(values, axis=1)
            aspect = sag_aspect
            position = position_3d[1]
        elif plane == 'axial':
            values = np.transpose(values, [2, 0, 1])
            aspect = trans_aspect
            position = position_3d[0]

        values, _ = trim_zeros(values[position])
        values = np.squeeze(values)

        if limits is None:
            limits = [0,  np.percentile(values, 99)]

        im1 = ax1.imshow(values, vmin=limits[0], vmax=limits[1],
                         cmap=cmap, aspect=aspect)
        if patch:
            for i in range(len(patch)):
                x_coord = patch[i][0][0]
                x_size = patch[i][1]
                y_coord = patch[i][0][1]
                y_size = patch[i][2]
                values_rect = values[y_coord:y_coord+y_size, x_coord:x_coord+x_size]
                val_patch.append((np.mean(values_rect), np.std(values_rect)))
                rect = Rectangle(patch[i][0],x_size,y_size,linewidth=2,edgecolor='r',facecolor='none')
                ax1.add_patch(rect)
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        plt.savefig(f'./figures/{fig_name}/{filename}_{plane}.eps')
        plt.show()

    if plot_cmap:
        fig, ax1 = plt.subplots(1, 1, figsize=(3,2))

        im1 = ax1.imshow(values, vmin=limits[0], vmax=limits[1], cmap=cmap,
                         aspect=aspect)
        im1.set_visible(False)
        plt.axis('off')
        cbar = plt.colorbar(im1, ax=ax1,  orientation="horizontal")
        plt.savefig(f'./figures/{fig_name}/{filename}_cmap.eps')
        plt.close()
    return val_patch


def plot_performance(processing_times, echoes='6 echoes', fig_name=''):
    fig, ax = plt.subplots(figsize=(4*1.5,2*1.5))

    methods = ['breast', 'single-res', 'breast', 'multi-res', 'single-res']
    tags = ['silicone', 'silicone', 'no silicone', 'no silicone', 'no silicone']
    x = np.arange(len(methods))

    mean = {}
    std = {}
    for device in processing_times[tags[0]].keys():
        mean[device] = []
        std[device] = []
        for i in range(len(methods)):
            values = processing_times[tags[i]][device][echoes][methods[i]]
            if len(values) == 0:
                mean[device].append(np.nan)
                std[device].append(np.nan)
            else:
                mean[device].append(np.mean(values))
                std[device].append(np.std(values))
        mean[device] = np.array(mean[device]) / 60
        std[device] = np.array(std[device]) / 60

        width = 0.35
        if device == 'iMac':
            ax.bar(x - width/2, mean[device], width, yerr=std[device], label='Desktop')
        else:
            ax.bar(x + width/2, mean[device], width, yerr=std[device], label='Workstation')

    plt.style.use('default')
    ax.set_ylabel('Time (min)')
    ax.set_xticks(x)
    ax.set_xticklabels(['hmrGC-wfs', 'vlGC', 'hmrGC-wfs', 'hmrGC-wf', 'vlGC'])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend()

    fig.tight_layout()
    filename = f'performance_{echoes}'
    plt.savefig(f'./figures/{fig_name}/{filename}.eps')
    plt.show()
