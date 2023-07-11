# fft2png, Generate spectrum from audio to png
# Copyright (C) 2016  Olivier Jolly
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import argparse
import logging
import math
import sys

import numpy as np
import os
from PIL import Image
from PIL import ImageDraw
from multiprocessing import Process

from . import wavfile

__all__ = []
__version__ = 0.5
__date__ = '2015-12-18'
__updated__ = '2016-03-17'
__author__ = 'olivier@pcedev.com'

class SpectrumWriter(object):
    def __init__(self, opts):
        self.bucket_nb = opts.bar_count
        self.bucket_pixel_spacing = opts.bar_spacing
        self.bucket_pixel_width = opts.bar_width
        self.height = opts.image_height

        self.color = opts.color

        self.min_freq = opts.audio_min_freq
        self.max_freq = opts.audio_max_freq

        self.silence_ceiling = opts.silence_ceiling

        self.output_filename_mask = opts.output_filename_mask
        np.seterr(divide='ignore') 

    def write_bar(self, bucket_start, line_data):
        raise NotImplementedError

    def write_spectrum(self, frequencies, spectrum, frame_index):

        self.image = Image.new("RGBA",
                               (self.bucket_nb * (self.bucket_pixel_spacing + self.bucket_pixel_width), self.height))
        self.draw = ImageDraw.Draw(self.image)

        spectrum_len = len(spectrum)

        # on last frame, it might happen that len is 0, just exit in this case
        if spectrum_len == 0:
            return

        display_freq = np.logspace(np.log(self.min_freq) / np.log(3),
                                   np.log(self.max_freq) / np.log(3),
                                   self.bucket_nb, base=3)

        interpolated_spectrum = np.interp(display_freq, frequencies, spectrum, left=0, right=0)

        for bucket_idx in range(self.bucket_nb):

            try:
                attenuation = 10 * np.log(interpolated_spectrum[bucket_idx] / spectrum_len)
                line_data = max(self.height + (self.height * attenuation / self.silence_ceiling), 1)
                logging.debug("freq %f, %f dB", display_freq[bucket_idx], attenuation)
            except ValueError:
                # on last frame, we might not have enough data to compute the average in a bucket
                continue

            if line_data:
                bucket_start = bucket_idx * (self.bucket_pixel_spacing + self.bucket_pixel_width)
                self.write_bar(bucket_start, line_data)
                # draw.rectangle((bucket_start, height, bucket_start + bucket_pixel_width, - line_data), fill=(255, 255, 255))

        self.image.save(self.output_filename_mask.format(frame_index))

        del self.draw


class FilledRectangleSpectrumWriter(SpectrumWriter):
    def write_bar(self, bucket_start, line_data):
        self.draw.rectangle(
            (bucket_start, self.height, bucket_start + self.bucket_pixel_width, self.height - line_data),
            fill=self.color)


class HollowRectangleSpectrumWriter(SpectrumWriter):
    def write_bar(self, bucket_start, line_data):
        self.draw.rectangle(
            (bucket_start, self.height, bucket_start + self.bucket_pixel_width, self.height - line_data),
            outline=self.color)


class SymetricalFilledRectangleSpectrumWriter(SpectrumWriter):
    def write_bar(self, bucket_start, line_data):
        self.draw.rectangle((bucket_start, self.height / 2 + line_data / 2, bucket_start + self.bucket_pixel_width,
                             self.height / 2 - line_data / 2), fill=self.color)


class SymetricalHollowRectangleSpectrumWriter(SpectrumWriter):
    def write_bar(self, bucket_start, line_data):
        self.draw.rectangle((bucket_start, self.height / 2 + line_data / 2, bucket_start + self.bucket_pixel_width,
                             self.height / 2 - line_data / 2), outline=self.color)


RENDERERS = [FilledRectangleSpectrumWriter,
             HollowRectangleSpectrumWriter,
             SymetricalFilledRectangleSpectrumWriter,
             SymetricalHollowRectangleSpectrumWriter,
             ]

def smooth_spectrum(spectrum, previous_spectrum, alpha=0):
    try:
        return (spectrum + alpha * previous_spectrum) / (1 + alpha)
    except (ValueError, TypeError):
        return spectrum


def compute_frequencies(spectrum, fs):
    return np.arange(spectrum.size) * (fs / 2) / (spectrum.size - 1)


def main(argv=None):
    program_name = os.path.basename(sys.argv[0])
    program_version = "v" + str(__version__)
    program_build_date = "%s" % __updated__

    program_version_string = '%%prog %s (%s)' % (program_version, program_build_date)
    program_longdesc = '''Convert audio file to stack of images'''
    program_license = "GPL v3+ 2015 Olivier Jolly"

    if argv is None:
        argv = sys.argv[1:]

    try:
        parser = argparse.ArgumentParser(epilog=program_longdesc,
                                         description=program_license)
        parser.add_argument("-d", "--debug", dest="debug", action="store_true",
                            default=False,
                            help="debug operations [default: %(default)s]")
        parser.add_argument("-n", dest="dry_run", action="store_true",
                            default=False,
                            help="don't perform actions [default: %(default)s]"
                            )
        parser.add_argument("-r", "--framerate", dest="target_fps", default=30, type=float,
                            help="output framerate [default: %(default)s]")
        parser.add_argument("-R", "--renderer", dest="renderer", default=0, type=int, choices=range(len(RENDERERS)),
                            help="which renderer to use to display bars (0=filled, 1=hollow, "
                                 "2=symetrical filled, 3=symetrical hollow)")
        parser.add_argument("-v", "--version", action="version", version=program_version_string)

        parser.add_argument("-w", "--bar-width", dest="bar_width", default=5, type=int,
                            help="bar width in output images")
        parser.add_argument("-s", "--bar-spacing", dest="bar_spacing", default=2, type=int,
                            help="bar spacing in output images")
        parser.add_argument("-c", "--bar-count", dest="bar_count", default=100, type=int,
                            help="number of bars in output images")
        parser.add_argument("-C", "--color", dest="color", default='FFFFFFFF', type=str,
                            help="hexa color of bars [RRGGBB or RRGGBBAA, default: %(default)s]")
        parser.add_argument("-b", "--blending", dest="blending", default=0.7, type=float,
                            help="blending of previous spectrum into current one "
                                 "(0 = display only fresh data, 1 = use as many previous than fresh data)")
        parser.add_argument("-W", "--window", dest="fft_window", default=4096, type=int,
                            help="window size for FFT [default: %(default)s]")

        parser.add_argument("--image-height", dest="image_height", default=120, type=int, help="output images height")

        parser.add_argument("--audio-min-freq", dest="audio_min_freq", default=50, type=int,
                            help="min frequency in input audio")
        parser.add_argument("--audio-max-freq", dest="audio_max_freq", default=2500, type=int,
                            help="max frequency in input audio")
        parser.add_argument("--silence-ceiling", dest="silence_ceiling", default=70, type=float,
                            help="opposite of threshold considered silence [in dB, default: %(default)s]")

        parser.add_argument("-i", dest="input_filename", default="input.wav", help="input file in wav format")
        parser.add_argument("-o", dest="output_filename_mask", required=True,
                            help="output filename mask (should contain {:06} or similar to generate sequence)")
        parser.add_argument("--n-cpus", dest="cpus_to_use", default=1, type=int,
                            help="number of cpus to use for wave generation")

        # process options
        opts = parser.parse_args(argv)

    except Exception as e:
        indent = len(program_name) * " "
        sys.stderr.write(program_name + ": " + repr(e) + "\n")
        sys.stderr.write(indent + "  for help use --help")
        return 2

    if opts.debug:
        logging.root.setLevel(logging.DEBUG)
    else:
        logging.root.setLevel(logging.INFO)

    # convert hexa in input to RGBA tuple
    opts.color = (
        int(opts.color[0:2], 16),
        int(opts.color[2:4], 16),
        int(opts.color[4:6], 16),
        int(opts.color[6:8], 16) if len(opts.color) > 6 else 255
    )

    # load input file
    fs, data = wavfile.read(opts.input_filename)

    # compute bits per sample
    normalize_bits = 8 * data.dtype.itemsize - 1

    # monoize potential stereo files
    if len(data.shape) > 1:
        raw_data = data.T[0] + data.T[1]
        normalize_bits += 1
    else:
        raw_data = data.T

    # and normalize audio file in the [ -1 ; 1 ] range
    normalized_data = raw_data / 2 ** normalize_bits

    byte_per_frame = fs / opts.target_fps

    frame_index_max = math.ceil(len(normalized_data) / byte_per_frame)
    frame_index_slice = math.floor(frame_index_max / opts.cpus_to_use)
    frame_index_module = frame_index_max % opts.cpus_to_use

    proc_list = []
    for i in range(opts.cpus_to_use):
        frame_i = i * frame_index_slice
        frame_j = (i + 1) * frame_index_slice

        last_slice = i == range(opts.cpus_to_use)[-1]
        # Add the frames that did not fit in any slice to the last slice
        frame_j = frame_j + frame_index_module if last_slice else frame_j

        p = Process(target=generate_waves, args=(normalized_data, byte_per_frame, frame_i, frame_j, fs, opts, last_slice))
        p.start()
        proc_list.append(p)

    for p in proc_list:
        p.join()

def generate_waves(normalized_data, byte_per_frame, frame_index, frame_index_max, fs, opts, verbose):
    previous_spectrum = None
    frame_init = frame_index
    writer = RENDERERS[opts.renderer](opts)

    while frame_index < frame_index_max:
        if verbose:
            print("{0:6}/{1:6}".format(frame_index - frame_init, frame_index_max - frame_init), end="\r")

        # compute the raw spectrum
        frame_start = frame_index * int(byte_per_frame)
        spectrum = abs(np.fft.rfft(normalized_data[frame_start:frame_start + opts.fft_window]))

        # smooth it over time
        spectrum = smooth_spectrum(spectrum, previous_spectrum, opts.blending)

        # compute frequencies
        frequencies = compute_frequencies(spectrum, fs)

        # write spectrum to file
        writer.write_spectrum(frequencies, spectrum, frame_index)

        frame_index += 1
        previous_spectrum = spectrum

    return 0


if __name__ == "__main__":
    sys.exit(main())
