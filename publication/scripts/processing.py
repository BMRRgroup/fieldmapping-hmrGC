from hmrGC.dixon_imaging import MultiEcho
import numpy as np
import h5py
import json
import time
import copy
import os

with open('../helper/fat_model_invivo.json') as f:
    fat_model = json.load(f)
with open('../helper/silicone_model.json') as f:
    silicone_model = json.load(f)

for root, dirs, files in os.walk('../data/source/'):
    for source_file in files:
        print(source_file)
        try:
            source = h5py.File(f'../data/source/{source_file}', 'r')
            subject = source_file.split('_')[1]
            processing = h5py.File(f'../data/processed_iMac/processed_{subject}', 'w')

            signal = source['multiecho']
            params = {}
            params.update(source['multiecho'].attrs)

            params.update(silicone_model)
            params['FatModel'] = fat_model
            mip = np.sqrt(np.sum(np.abs(signal)**2, axis=-1))
            mask = mip > 7.5 / 100 * np.max(mip)

            methods = ['breast', 'multi-res', 'single-res']
            echoes = [6, 4]
            for num_echoes in echoes:
                params['TE_s'] = params['TE_s'][:num_echoes]
                signal = signal[:,:,:,:num_echoes]
                grp = processing.create_group(f'{num_echoes} echoes')

                for method in methods:
                    g = MultiEcho(signal, mask, params)
                    if method != 'breast' and subject[-4] != 'S':
                        g.params['signal_model'] = 'WF'
                        g.set_options_default()
                    g.range_fm *= 2.5
                    g.r2star_correction = False

                    start = time.time()
                    g.perform(method)
                    end = time.time()

                    grp_method = grp.create_group(method)
                    grp_method.attrs['processing time'] = end - start

                    save_params = ['water', 'fat', 'fatFraction_percent',
                                   'fieldmap', 'r2starmap']
                    if 'silicone' in g.images.keys():
                        save_params.append('silicone')

                    for save_param in save_params:
                        if save_param == 'fieldmap':
                            param = grp_method.create_dataset(f'{save_param}', data=g.fieldmap,
                                                    compression='gzip')
                        elif save_param == 'r2starmap':
                            param = grp_method.create_dataset(f'{save_param}', data=g.r2starmap,
                                                    compression='gzip')
                        else:
                            param = grp_method.create_dataset(f'{save_param}', data=g.images[f'{save_param}'],
                                                    compression='gzip')


            processing.close()
        except OSError:
            continue
