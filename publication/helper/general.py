import numpy as np
import sys
import h5py as h5
sys.path.insert(0, './helper/MR_CSS/pycss')
import pycss.css as css
import pycss.wfi as wfi
from pycss.Fatmodel import Fatmodel
from scipy.ndimage import binary_fill_holes, label


def add_rician_noise(refsignal, SNR):
    sigma_noise_mag = np.max(np.abs(refsignal[..., 0])) / SNR

    noiseReal = sigma_noise_mag * \
        np.random.randn(*refsignal.shape).astype(np.float32)
    noiseImaginary = sigma_noise_mag * \
        np.random.randn(*refsignal.shape).astype(np.float32)
    noiseComplex = noiseReal + 1.0j * noiseImaginary
    signal = refsignal + noiseComplex
    return signal


def css_silicone(in_params, options):
    in_params = in_params.copy()
    options = options.copy()

    TE_s = in_params['TE_s'].ravel()

    signal = in_params['signal']
    mask = options['mask']
    init_fieldmap_Hz = options['init_fieldmap_Hz']

    F = Fatmodel(modelname = options.get('fatmodel', 'Ren marrow'))
    chemicalShiftSilicone = np.array([-F.centerfreq_Hz * 1e-6 * 4.9])
    chemicalShiftsWaterFat = F.get_chemical_shifts_Hz()
    chemicalShifts = np.concatenate(([chemicalShiftsWaterFat[0]],
                                     chemicalShiftSilicone,
                                     chemicalShiftsWaterFat[1:]))

    nFat = len(F.deshielding_ppm)
    shape = (nFat + 2, 3)
    shape2 = (nFat + 2, 1)
    Cm = np.zeros(shape)
    Cm[0, 0] = 1
    Cm[1, 1] = 1
    Cm[2:nFat+2, 2] = F.relamps_percent[::] / 100

    Cp = Cm != 0
    Cp = Cp.astype(np.float)

    Cr = np.ones(shape2)
    Cf = np.ones(shape2)

    Pm0 = wfi.build_Pm0(chemicalShifts, init_fieldmap_Hz[mask].ravel())

    default_options = {'mask': mask,
                       'Cm': Cm,
                       'Cp': Cp,
                       'Cf': Cf,
                       'Cr': Cr,
                       'Pm0': Pm0,
                       'tol': options['tol'],
                       'itermax': 100,
                       'verbose': True}

    default_options.update(options)
    options = default_options

    outParams = css.css_varpro(in_params, options)

    model_dict = wfi.structure_param_maps(outParams['param_maps'],
                                       [options['Cm'],
                                       options['Cp'],
                                       options['Cf'],
                                       options['Cr']])
    outParams.update(model_dict)
    outParams.update(options)
    return outParams


def get_tissueMaskFilled(signal, threshold):
    mip = np.sqrt(np.sum(np.abs(signal)**2, axis=-1))
    tissueMask = mip > threshold / 100 * np.max(mip)

    label_objects, _ = label(tissueMask)
    sizes = np.bincount(label_objects.ravel())[1:]
    helper_mask = (label_objects == np.argmax(sizes)+1)
    tmp_filledMask = np.zeros_like(tissueMask)
    tmp_filledMask[helper_mask] = tissueMask[helper_mask]
    tissueMask = tmp_filledMask

    filledMask = np.zeros_like(tissueMask)
    for ie in range(0, tissueMask.shape[-1]):
        filledMask[:,:,ie] = binary_fill_holes(tissueMask[:,:,ie])
    return filledMask


def get_BMRS_quantification(BMRS_obj_path):
    quant = {}
    f = h5.File(BMRS_obj_path, 'r')
    ref = h5.File(BMRS_obj_path, 'r')['BMRSobj']['quant'][0, 0]
    data = f[ref]['output']['param']
    for key in data.keys():
        quant[key] = np.squeeze(np.array(data[key]))
    return quant
