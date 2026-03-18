"""
Compression function benchmarking

Author: Sam Barba
Created 09/03/2026
"""

from compression import bz2, gzip, lzma, zlib, zstd
import random
import string
from time import perf_counter

import matplotlib.pyplot as plt


def benchmark_compression(compress_fn, decompress_fn, data):
	start = perf_counter()
	compressed = compress_fn(data)
	compress_time = round(1000 * (perf_counter() - start))

	start = perf_counter()
	decompressed = decompress_fn(compressed)
	decompress_time = round(1000 * (perf_counter() - start))

	assert decompressed == data
	compressed_size_proportion = len(compressed) / len(data)

	return compress_time, decompress_time, compressed_size_proportion


if __name__ == '__main__':
	# Test string compression

	data = ''.join(random.choice(string.printable) for _ in range(5_000_000)).encode()

	names = []
	compress_times = []
	decompress_times = []
	compression_proportions = []

	for mod in [bz2, gzip, lzma, zlib, zstd]:
		compress_fn = getattr(mod, 'compress')
		decompress_fn = getattr(mod, 'decompress')
		compress_time, decompress_time, proportion = benchmark_compression(compress_fn, decompress_fn, data)
		names.append(mod.__name__.removeprefix('compression.'))
		compress_times.append(compress_time)
		decompress_times.append(decompress_time)
		compression_proportions.append(proportion)

	_, axes = plt.subplots(nrows=3, figsize=(10, 7), layout='tight')

	axes[0].set_title('Compression time (ms)', fontsize=11)
	bars0 = axes[0].bar(names, compress_times)
	axes[0].bar_label(bars0)
	axes[0].set_ylim(0, max(compress_times) * 1.2)

	axes[1].set_title('Decompression time (ms)', fontsize=11)
	bars1 = axes[1].bar(names, decompress_times)
	axes[1].bar_label(bars1)
	axes[1].set_ylim(0, max(decompress_times) * 1.2)

	axes[2].set_title('% size of original', fontsize=11)
	bars2 = axes[2].bar(names, compression_proportions)
	labels = [f'{x:.2%}' for x in compression_proportions]
	axes[2].bar_label(bars2, labels=labels)
	axes[2].set_ylim(min(compression_proportions) * 0.998, max(compression_proportions) * 1.002)

	plt.suptitle('String compression test', fontsize=14, x=0.52)
	plt.show()

	# Test file compression

	file_path = ('C:/Users/sam/Desktop/projects/python/_artificial_intelligence_machine_learning'
		'/pytorch_autoencoder_variational/model.pth')
	with open(file_path, 'rb') as f:
		file_data = f.read()

	compress_times = []
	decompress_times = []
	compression_proportions = []

	for mod in [bz2, gzip, lzma, zlib, zstd]:
		compress_fn = getattr(mod, 'compress')
		decompress_fn = getattr(mod, 'decompress')
		compress_time, decompress_time, proportion = benchmark_compression(compress_fn, decompress_fn, file_data)
		compress_times.append(compress_time)
		decompress_times.append(decompress_time)
		compression_proportions.append(proportion)

	_, axes = plt.subplots(nrows=3, figsize=(10, 7), layout='tight')

	axes[0].set_title('Compression time (ms)', fontsize=11)
	bars0 = axes[0].bar(names, compress_times)
	axes[0].bar_label(bars0)
	axes[0].set_ylim(0, max(compress_times) * 1.2)

	axes[1].set_title('Decompression time (ms)', fontsize=11)
	bars1 = axes[1].bar(names, decompress_times)
	axes[1].bar_label(bars1)
	axes[1].set_ylim(0, max(decompress_times) * 1.2)

	axes[2].set_title('% size of original', fontsize=11)
	bars2 = axes[2].bar(names, compression_proportions)
	labels = [f'{x:.2%}' for x in compression_proportions]
	axes[2].bar_label(bars2, labels=labels)
	axes[2].set_ylim(min(compression_proportions) * 0.99, max(compression_proportions) * 1.01)

	plt.suptitle('File compression test', fontsize=14, x=0.526)
	plt.show()
