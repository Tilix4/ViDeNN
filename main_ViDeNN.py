# -*- coding: utf-8 -*-
"""
@author: clausmichele
"""

import argparse
from pathlib import Path
import tensorflow as tf
import os

from model_ViDeNN import ViDeNN



def ViDeNNDenoise(ViDeNN: ViDeNN, sess, checkpoint_dir: Path, source_dir: Path, img_format: str, export_dir: Path):
	eval_files_noisy = source_dir.glob(f"*.{img_format}")
	eval_files_noisy = sorted(eval_files_noisy)
	eval_files_noisy = [f.as_posix() for f in eval_files_noisy]
	eval_files = []
	print_psnr = True
	if eval_files == []:
		eval_files = eval_files_noisy
		print_psnr = False
		print("[*] No original frames found, not printing PSNR values...")
	eval_files = sorted(eval_files)
	ViDeNN.denoise(sess, eval_files, eval_files_noisy, print_psnr, checkpoint_dir.as_posix(), export_dir.as_posix()) # TODO refactor all with pathlib

def main(_):
	parser = argparse.ArgumentParser(description='')
	parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
	parser.add_argument('--save_dir', dest='save_dir', default=Path('./data/denoised'), help='denoised sample are saved here')
	parser.add_argument('--test_dir', dest='test_dir', default='./data', help='directory of noisy frames')
	parser.add_argument('--img_format', dest='img_format', default='png', help='denoised sample are saved here')
	parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default=None, help='path of ViDeNN checkpoint')
	args = parser.parse_args()

	source_dir = Path(args.test_dir)
	export_dir = Path(args.save_dir).resolve()
	if not export_dir.is_dir():
		export_dir.mkdir(parents=True)
	if args.use_gpu:
		# added to control the gpu memory
		print("GPU\n")
		gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
		with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
			model = ViDeNN(sess)
	else:
		print("CPU\n")
		with tf.device('/cpu:0'):
			with tf.Session() as sess:
				model = ViDeNN(sess)

	ViDeNNDenoise(model, sess, Path(args.ckpt_dir), source_dir, args.img_format, export_dir)


if __name__ == '__main__':
	tf.app.run()
