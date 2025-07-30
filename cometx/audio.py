# -*- coding: utf-8 -*-
# ****************************************
#                              __
#   _________  ____ ___  ___  / /__  __
#  / ___/ __ \/ __ `__ \/ _ \/ __/ |/_/
# / /__/ /_/ / / / / / /  __/ /__>  <
# \___/\____/_/ /_/ /_/\___/\__/_/|_|
#
#
#  Copyright (c) 2024 Cometx Development
#      Team. All rights reserved.
# ****************************************

import io
import os
import random

import matplotlib.pyplot as plt
import numpy
from comet_ml.convert_utils import write_numpy_array_as_wav
from scipy.io import wavfile


def log_spectrogram(experiment, channel, sample_rate, title, outname, step):
    """
    Logs a spectrogram figure to the experiment.
    """
    plt.figure(figsize=(12, 6))
    plt.specgram(channel, Fs=sample_rate, vmin=-20, vmax=50)
    plt.title(title)
    plt.ylabel("Frequency (Hz)")
    plt.xlabel("Time (s)")
    plt.colorbar()
    plt.savefig(outname + "-spectrogram.png")
    plt.close()
    experiment.log_image(outname + "-spectrogram.png", step=step)


def log_waveform(experiment, data, title, outname, step):
    """
    Logs a waveform figure to the experiment.
    """
    plt.figure(figsize=(12, 6))
    if len(data.shape) == 1:
        plt.plot(data)
    else:
        for channel in range(data.shape[1]):
            plt.plot(data[:, channel], label=f"Channel {channel+1}")
        plt.legend()
    plt.title(title)
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.savefig(outname + "-waveform.png")
    plt.close()
    experiment.log_image(outname + "-waveform.png", step=step)


def log_audio(
    experiment,
    audio_data,
    sample_rate=None,
    file_name=None,
    metadata=None,
    overwrite=False,
    copy_to_tmp=True,
    step=None,
):
    """
    Logs the audio, wavform figure, and spectrogram figure
    to the experiment. Same signature as
    comet_ml.Experiment.log_audio().

    NOTE: This assumes either a wav file, or numpy array.
    """
    if isinstance(audio_data, numpy.ndarray):
        audio_fp = io.BytesIO()
        write_numpy_array_as_wav(audio_data, sample_rate, audio_fp)
        audio_fp.seek(0)
        if file_name:
            basename, ext = os.path.splitext(os.path.basename(file_name))
        else:
            basename = "audio-%s" % random.randint(10000, 99999)

    elif isinstance(audio_data, str) and audio_data.endswith(".wav"):
        audio_fp = open(audio_data, "rb")
        if file_name:
            basename, ext = os.path.splitext(os.path.basename(file_name))
        else:
            basename, ext = os.path.splitext(os.path.basename(audio_data))
    else:
        raise Exception("Unable to handle this audio file format; " + "use .wav file")

    # Overwrite, get from file:
    sample_rate, data = wavfile.read(audio_fp)

    # Plot the Waveform
    log_waveform(experiment, data, "Waveform", basename, step)

    # Plot the Spectrogram
    if len(data.shape) == 1:
        log_spectrogram(
            experiment,
            data,
            sample_rate,
            "Mono Channel",
            basename,
            step,
        )
    else:
        for channel, title in zip(
            range(data.shape[1]), ["Left Channel", "Right Channel"]
        ):
            log_spectrogram(
                experiment,
                data[:, channel],
                sample_rate,
                title,
                basename + "-" + title.split(" ")[0].lower(),
                step,
            )

    return experiment.log_audio(
        audio_data,
        sample_rate=sample_rate,
        file_name=basename,
        metadata=metadata,
        overwrite=overwrite,
        copy_to_tmp=copy_to_tmp,
        step=step,
    )
