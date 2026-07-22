"""
Adversarial Attacks on Pretrained ResNet18

Author: Sam Barba
Created 2026-06-16
"""

import tkinter as tk

import torch

from _utils.plotting import plot_saliency_map
from utils import *


BATCH_SIZE = 4


def print_row(col1, col2, col3, col4):
	def fmt(x):
		return f'{x:.3f}' if isinstance(x, float) else x

	print(f'{fmt(col1):<47} | {fmt(col2):^19} | {fmt(col3):^14} | {fmt(col4):^14}')


if __name__ == '__main__':
	plot_learned_patches(['untargeted'] + CLASS_SUBSET[:5])
	plot_learned_patches(CLASS_SUBSET[5:])

	# Load data

	*_, test_loader = create_data_loaders(BATCH_SIZE)
	test_loader_iter = iter(test_loader)

	# Plot saliency maps for some clean images

	model.eval()
	for idx, (x, _) in enumerate(test_loader):
		x = x.to(DEVICE)
		for img in x:
			plot_saliency_map(model, img, imagenet_labels, pil_image_transform)
		if idx > 3:
			break

	# Print evaluation results for each attack type (inc. clean input)

	stats = dict()
	print(f"\n{'Attack type':^47} | {'Attack success rate':^19} | {'Top-1 accuracy':^14} | {'Top-5 accuracy':^14}")
	print('-' * 48 + '|' + '-' * 21 + '|' + '-' * 16 + '|' + '-' * 16)

	# Evaluate model (no adversarial attacks)

	top1_acc, top5_acc = evaluate(model, test_loader)
	print_row('None', '-', top1_acc, top5_acc)
	stats['None'] = (top1_acc, top5_acc)

	# Evaluate model with patch attacks

	for c in ['untargeted'] + CLASS_SUBSET:
		cl = clean_label(c)
		for patch_size in PATCH_SIZES:
			patch_logits = torch.load(f'./artefacts/patch_{c}_{patch_size}x{patch_size}.pth', map_location=DEVICE)
			attack_success_rate, top1_acc, top5_acc = evaluate(
				model,
				test_loader,
				patch_logits=patch_logits,
				target_class_name=None if cl == 'untargeted' else cl
			)
			key = f'{patch_size}x{patch_size} patch (' + ('untargeted)' if cl == 'untargeted' else f"target class = '{cl}')")
			print_row(key, attack_success_rate, top1_acc, top5_acc)
			stats[key] = (attack_success_rate, top1_acc, top5_acc)

	# Evaluate model with perturbation attacks

	for c in ['untargeted'] + CLASS_SUBSET:
		perturbation = torch.load(f'./artefacts/perturbation_{c}.pth', map_location=DEVICE)
		c = clean_label(c)
		attack_success_rate, top1_acc, top5_acc = evaluate(
			model,
			test_loader,
			perturbation=perturbation,
			target_class_name=None if c == 'untargeted' else c
		)
		key = 'Perturbation (' + ('untargeted)' if c == 'untargeted' else f"target class = '{c}')")
		print_row(key, attack_success_rate, top1_acc, top5_acc)
		stats[key] = (attack_success_rate, top1_acc, top5_acc)

	# Create Tkinter UI for iterating over test loader

	root = tk.Tk()
	root.title('Adversarial attack demo')
	root.config(width=380, height=180, background='#101010')
	root.resizable(False, False)

	attack_type_lbl = tk.Label(root, text='Attack type:',
		font='consolas 10', background='#101010', foreground='white')
	stringv_attack_type = tk.StringVar(value='none')
	radio_btn_none = tk.Radiobutton(root, text='None', font='consolas 10', variable=stringv_attack_type,
		value='none', background='#101010', foreground='white',
		activebackground='#101010', activeforeground='white', selectcolor='#101010')
	radio_btn_patch = tk.Radiobutton(root, text='Patch', font='consolas 10', variable=stringv_attack_type,
		value='patch', background='#101010', foreground='white',
		activebackground='#101010', activeforeground='white', selectcolor='#101010')
	radio_btn_perturbation = tk.Radiobutton(root, text='Perturbation', font='consolas 10', variable=stringv_attack_type,
		value='perturbation', background='#101010', foreground='white',
		activebackground='#101010', activeforeground='white', selectcolor='#101010')

	patch_size_lbl = tk.Label(root, text='Patch size:',
		font='consolas 10', background='#101010', foreground='white')
	intv_patch_size = tk.IntVar(value=PATCH_SIZES[0])
	patch_size_option_menu = tk.OptionMenu(root, intv_patch_size, *PATCH_SIZES)
	patch_size_option_menu.config(font='consolas 10', indicatoron=0, highlightthickness=0)

	target_class_lbl = tk.Label(root, text='Target class:',
		font='consolas 10', background='#101010', foreground='white')
	stringv_target_class = tk.StringVar(value='untargeted')
	cleaned_labels = ['untargeted'] + [clean_label(i) for i in CLASS_SUBSET]
	target_class_option_menu = tk.OptionMenu(root, stringv_target_class, *cleaned_labels)
	target_class_option_menu.config(font='consolas 10', indicatoron=0, highlightthickness=0)

	def next_test_batch():
		global test_loader_iter

		try:
			test_batch = next(test_loader_iter)
		except StopIteration:
			test_loader_iter = iter(test_loader)
			test_batch = next(test_loader_iter)

		attack_type = stringv_attack_type.get()
		patch_size = intv_patch_size.get()
		target_class = stringv_target_class.get()
		target_class_pth = target_class.replace(' ', '_')

		patch_logits = perturbation = None
		top1_acc, top5_acc = stats['None']
		title_append = f'\nTop-1 acc: {top1_acc:.3f}  |  Top-5 acc: {top5_acc:.3f}'

		if attack_type == 'patch':
			patch_logits = torch.load(f'./artefacts/patch_{target_class_pth}_{patch_size}x{patch_size}.pth', map_location=DEVICE)
			key = f'{patch_size}x{patch_size} patch (' + ('untargeted)' if target_class == 'untargeted' else f"target class = '{target_class}')")
			asr, top1_acc, top5_acc = stats[key]
			title_append = f'\nAttack success rate: {asr:.3f}  |  Top-1 acc: {top1_acc:.3f}  |  Top-5 acc: {top5_acc:.3f}'
		elif attack_type == 'perturbation':
			perturbation = torch.load(f'./artefacts/perturbation_{target_class_pth}.pth', map_location=DEVICE)
			key = 'Perturbation (' + ('untargeted)' if target_class == 'untargeted' else f"target class = '{target_class}')")
			asr, top1_acc, top5_acc = stats[key]
			title_append = f'\nAttack success rate: {asr:.3f}  |  Top-1 acc: {top1_acc:.3f}  |  Top-5 acc: {top5_acc:.3f}'

		plot_output(
			model,
			test_batch,
			patch_logits=patch_logits,
			perturbation=perturbation,
			target_class_name=None if target_class == 'untargeted' else target_class,
			title_append=title_append
		)

	btn_sample_batch = tk.Button(root, text='Next batch', font='consolas 11', command=next_test_batch)

	attack_type_lbl.place(width=100, height=29, x=13, y=10)
	radio_btn_none.place(width=60, height=29, x=108, y=10)
	radio_btn_patch.place(width=65, height=29, x=174, y=10)
	radio_btn_perturbation.place(width=110, height=29, x=248, y=10)
	patch_size_lbl.place(width=100, height=29, x=17, y=45)
	patch_size_option_menu.place(width=50, height=29, x=115, y=47)
	target_class_lbl.place(width=100, height=29, x=10, y=80)
	target_class_option_menu.place(width=130, height=29, x=115, y=82)
	btn_sample_batch.place(width=150, height=29, x=115, y=130)

	next_test_batch()

	root.mainloop()
