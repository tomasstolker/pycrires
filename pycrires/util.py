import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

def lowpass_filter(flux, window_length):
    nans = np.isnan(flux)
    if np.sum(nans)>0.5*flux.size:
        return np.tile(np.nan, flux.size)
    else:
        window_length = min(window_length, 2*(np.sum(~nans)//2)-1)
        filtered = flux.copy()
        filtered[~nans] = np.array(savgol_filter(flux[~nans],
                        window_length, 2))
        return filtered

def highpass_filter(order_flux, window_length):
    continuum = lowpass_filter(order_flux, window_length)
    return order_flux - continuum

def mask_tellurics(order_flux, order_wl, lower_lim = 0.5, upper_lim=2.):
    wl = np.nanmedian(order_wl, axis=0)
    tot_spec = np.nansum(order_flux, axis=0)
    percentile = np.nanpercentile(tot_spec, 0.3)
    mask = tot_spec>percentile
    continuum = np.poly1d(np.polyfit(wl[mask], tot_spec[mask], 3))(wl)
    cont_normalized_flux = tot_spec/continuum
    to_mask = (cont_normalized_flux<lower_lim) + (cont_normalized_flux>upper_lim)
    masked_order = np.copy(order_flux)
    masked_order[:,to_mask] = np.nan
    return masked_order

def fit_svd_kernel(order_flux, order_wl, star_model, max_shift=50, rcond=1e-3):
    result = np.copy(order_flux)
    for i, (spec, wl, model) in enumerate(zip(order_flux, order_wl, star_model)):
        interpolator = interp1d(wl, model, bounds_error=False,
            fill_value=np.nanmedian(model))
        shifts = np.arange(-max_shift, max_shift+1e-10)
        dlam = np.nanmean(np.diff(wl))
        modes = [interpolator(wl+shift*dlam) for shift in shifts]
        modes = np.array(modes)
        modes[np.isnan(modes)] = 0
        proj_matrix = np.linalg.pinv(modes, rcond=rcond)
        spec[np.isnan(spec)] = 0
        amps = proj_matrix.T.dot(spec)
        reconstructed = amps.dot(modes)
        result[i] = reconstructed
    return result

def flag_outliers(order_spec, sigma=4):
    z = (order_spec - np.nanmedian(order_spec, axis=1)[:, np.newaxis])\
                /np.nanstd(order_spec, axis=1)[:, np.newaxis]
    outliers = z>sigma
    order_spec[outliers] = np.nan
    return order_spec

