import numpy as np


def add_gaussian_white_noise(x, target_snr_db=0, target_noise_db=0, mode='noisePower'):
    '''
    Add additive gaussian white noise to signals based on the methods chosen.

    Parameters:
    -----------
    - x : ndarray
        shape: (time, ), a single channel EEG data.
    - mode : str
        'targetSNR'
        'noisePower'
    - target_snr_db : SNR for the target signal in dB. used when mode == 'targetSNR'
    - target_noise_db: the power of the noise in dB.

    :return:
    '''

    # power of the signal in watts
    x_watts = x ** 2


    if mode == 'targetSNR':
        # Adding noise using target SNR

        # Calculate signal power and convert to dB
        sig_avg_watts = np.mean(x_watts)
        sig_avg_db = 10 * np.log10(sig_avg_watts)
        # Calculate noise power then convert to watts
        noise_avg_db = sig_avg_db - target_snr_db
        noise_avg_watts = 10 ** (noise_avg_db / 10)
        # Generate a sample of white noise
        mean_noise = 0
        noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(x_watts))
    elif mode == 'noisePower':
        # Convert to linear Watt units
        target_noise_watts = 10 ** (target_noise_db / 10)
        # Generate noise samples
        mean_noise = 0
        noise_volts = np.random.normal(mean_noise, np.sqrt(target_noise_watts), len(x_watts))

    noisy_x = x + noise_volts

    return noisy_x