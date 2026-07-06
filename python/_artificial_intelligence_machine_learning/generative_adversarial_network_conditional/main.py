"""
PyTorch demo of a Conditional Generative Adversarial Network (cGAN)

Author: Sam Barba
Created 2026-07-06
"""

from pathlib import Path
import tkinter as tk

import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from _utils.custom_dataset import CustomDataset
from _utils.plotting import plot_image_grid
from _utils.progress_bar import ProgressBar
from models import Generator, Discriminator


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

DATASET_DICT = {
	'gender': ['male', 'female'],
	'race': ['white', 'black', 'asian', 'indian', 'other']
}
IMG_SIZE = 64
GEN_LATENT_DIM = 128
BATCH_SIZE = 64
GEN_LEARNING_RATE = 1e-4
DISC_LEARNING_RATE = 4e-4
DISC_STEPS = 2  # Train discriminator twice per generator step
OPTIM_BETAS = (0.0, 0.99)
NUM_EPOCHS = 100
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

destandardise_transform = transforms.Lambda(lambda img: img * 0.5 + 0.5)


def create_train_loader():
	transform = transforms.Compose([
		transforms.Resize(IMG_SIZE),
		transforms.ToTensor(),  # Scale to [0,1]
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

	img_paths = list(Path('C:/Users/sam/Desktop/projects/datasets/utkface').glob('*.jpg'))
	x = [
		transform(Image.open(p)) for p in
		ProgressBar(img_paths, desc='Preprocessing images', unit='imgs')
	]
	y_raw = [str(i).split('\\')[-1] for i in img_paths]
	y_split = [i.split('_')[:3] for i in y_raw]
	y_age_bin = [min(int(i[0]) // 10, 9) for i in y_split]
	y_gender_id = [int(i[1]) for i in y_split]
	y_race_id = [int(i[2]) for i in y_split]
	y = torch.tensor(list(zip(y_age_bin, y_gender_id, y_race_id)), dtype=torch.long)

	dataset = CustomDataset(x, y)
	train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

	return train_loader


def plot_output(age_bin, gender, race, save_path):
	gender_id = DATASET_DICT['gender'].index(gender)
	race_id = DATASET_DICT['race'].index(race)
	age_bin_tensor = torch.full((num_images, 1), age_bin, device=DEVICE, dtype=torch.long)
	gender_id_tensor = torch.full((num_images, 1), gender_id, device=DEVICE, dtype=torch.long)
	race_id_tensor = torch.full((num_images, 1), race_id, device=DEVICE, dtype=torch.long)
	labels = torch.cat([age_bin_tensor, gender_id_tensor, race_id_tensor], dim=1)

	with torch.inference_mode():
		fake = gen_model(fixed_noise, labels)
	plot_image_grid(
		fake, rows=4, cols=6, padding=4, transform=destandardise_transform, scale_factor=1.5,
		scale_interpolation='cubic', background_rgb=(0, 0, 0), title_rgb=(255, 255, 255),
		title=f'Generator test: age bin #{age_bin}, {gender}, {race}',
		save_path=save_path,
		show='epoch' not in save_path
	)


if __name__ == '__main__':
	gen_model = Generator(latent_dim=GEN_LATENT_DIM).to(DEVICE)

	# For generation
	num_images = 24
	fixed_noise = torch.randn(num_images, GEN_LATENT_DIM, 1, 1, device=DEVICE)

	if Path('./gen_model.pth').exists():
		gen_model.load_state_dict(torch.load('./gen_model.pth', map_location=DEVICE))
	else:
		disc_model = Discriminator().to(DEVICE)
		train_loader = create_train_loader()
		gen_optimiser = torch.optim.Adam(gen_model.parameters(), lr=GEN_LEARNING_RATE, betas=OPTIM_BETAS)
		disc_optimiser = torch.optim.Adam(disc_model.parameters(), lr=DISC_LEARNING_RATE, betas=OPTIM_BETAS)
		disc_model.train()

		print('\n----- TRAINING -----\n')

		for epoch in range(1, NUM_EPOCHS + 1):
			gen_model.train()

			prog_bar = ProgressBar(train_loader, desc=f'Epoch {epoch}/{NUM_EPOCHS}', unit='batch', auto_finish=False)
			total_disc_real_loss = total_disc_fake_loss = total_gen_loss = 0
			total_disc_real_score = total_disc_fake_score = 0

			for img_batch, labels in prog_bar:
				img_batch = img_batch.to(DEVICE)
				labels = labels.to(DEVICE)

				# Train discriminator

				for _ in range(DISC_STEPS):
					noise = torch.randn(img_batch.shape[0], GEN_LATENT_DIM, 1, 1, device=DEVICE)
					with torch.no_grad():
						fake = gen_model(noise, labels)

					disc_real = disc_model(img_batch, labels)
					disc_fake = disc_model(fake, labels)
					disc_real_loss = torch.relu(1 - disc_real).mean()
					disc_fake_loss = torch.relu(1 + disc_fake).mean()
					disc_loss = disc_real_loss + disc_fake_loss  # Hinge loss
					total_disc_real_loss += disc_real_loss.item()
					total_disc_fake_loss += disc_fake_loss.item()
					total_disc_real_score += disc_real.mean().item()
					total_disc_fake_score += disc_fake.mean().item()

					disc_optimiser.zero_grad()
					disc_loss.backward()
					disc_optimiser.step()

				# Train generator

				noise = torch.randn(img_batch.shape[0], GEN_LATENT_DIM, 1, 1, device=DEVICE)
				fake = gen_model(noise, labels)
				disc_fake = disc_model(fake, labels)
				gen_loss = -disc_fake.mean()
				total_gen_loss += gen_loss.item()

				gen_optimiser.zero_grad()
				gen_loss.backward()
				gen_optimiser.step()

			# To track output quality throughout training
			gen_model.eval()
			plot_output(age_bin=2, gender='male', race='white', save_path=f'./images/epoch_{epoch}.png')

			mean_disc_real_loss = total_disc_real_loss / (len(train_loader) * DISC_STEPS)
			mean_disc_fake_loss = total_disc_fake_loss / (len(train_loader) * DISC_STEPS)
			mean_disc_real_score = total_disc_real_score / (len(train_loader) * DISC_STEPS)
			mean_disc_fake_score = total_disc_fake_score / (len(train_loader) * DISC_STEPS)
			mean_gen_loss = total_gen_loss / len(train_loader)

			prog_bar.finish(
				f'D(real)={mean_disc_real_score:.2f}, '
				f'D(fake)={mean_disc_fake_score:.2f}, '
				f'L_real={mean_disc_real_loss:.4f}, '
				f'L_fake={mean_disc_fake_loss:.4f}, '
				f'L_G={mean_gen_loss:.4f}'
			)

			torch.save(gen_model.state_dict(), f'./gen_model_epoch_{epoch}.pth')

	gen_model.eval()

	# ----- Change these generator conditions ----- #
	age_bin = 2  # Max. 9
	gender = 'male'
	race = 'white'
	# --------------------------------------------- #

	# plot_output(age_bin, gender, race, f'./images/test_agebin_{age_bin}_gender_{gender}_race_{race}.png')

	# Interpolate between age bin 0 and 9

	# gender_id = DATASET_DICT['gender'].index('male')
	# race_id = DATASET_DICT['race'].index('white')
	# tmp = torch.empty(num_images, 1, device=DEVICE, dtype=torch.long)
	# gender_id_tensor = torch.full((num_images, 1), gender_id, device=DEVICE, dtype=torch.long)
	# race_id_tensor = torch.full((num_images, 1), race_id, device=DEVICE, dtype=torch.long)
	# labels = torch.cat([tmp, gender_id_tensor, race_id_tensor], dim=1)
	# embedding_0 = gen_model.age_embedding(torch.tensor([0], device=DEVICE))
	# embedding_9 = gen_model.age_embedding(torch.tensor([9], device=DEVICE))
	# for t in torch.linspace(0, 1, 101):
	# 	age_embedding_interp = t * embedding_9 + (1 - t) * embedding_0
	# 	age_embedding_interp = age_embedding_interp.repeat(num_images, 1)
	# 	with torch.inference_mode():
	# 		latent_space_test = gen_model(fixed_noise, labels, age_embedding=age_embedding_interp)
	# 	plot_image_grid(
	# 		latent_space_test, rows=4, cols=6, padding=4, transform=destandardise_transform, scale_factor=1.5,
	# 		scale_interpolation='cubic', background_rgb=(0, 0, 0), title_rgb=(255, 255, 255),
	# 		title=f'Age bin: {(9 * t):.2f}',
	# 		save_path=f'./images/age_interp_{t:.2f}.png',
	# 		show=False
	# 	)

	# Interpolate between 'black' and 'white'

	# age_tensor = torch.full((num_images, 1), 2, device=DEVICE, dtype=torch.long)
	# labels = torch.cat([age_tensor, gender_id_tensor, tmp], dim=1)
	# black_id = DATASET_DICT['race'].index('black')
	# white_id = DATASET_DICT['race'].index('white')
	# black_embedding = gen_model.race_embedding(torch.tensor([black_id], device=DEVICE))
	# white_embedding = gen_model.race_embedding(torch.tensor([white_id], device=DEVICE))
	# for t in torch.linspace(0, 1, 101):
	# 	race_embedding_interp = t * black_embedding + (1 - t) * white_embedding
	# 	race_embedding_interp = race_embedding_interp.repeat(num_images, 1)
	# 	with torch.inference_mode():
	# 		latent_space_test = gen_model(fixed_noise, labels, race_embedding=race_embedding_interp)
	# 	plot_image_grid(
	# 		latent_space_test, rows=4, cols=6, padding=4, transform=destandardise_transform, scale_factor=1.5,
	# 		scale_interpolation='cubic', background_rgb=(0, 0, 0), title_rgb=(255, 255, 255),
	# 		title=f'{t:.2f}(black) + {(1 - t):.2f}(white)',
	# 		save_path=f'./images/race_interp_{t:.2f}.png',
	# 		show=False
	# 	)

	# Create Tkinter UI for generating images

	def sample(*_, randomise_noise=False):
		global fixed_noise

		if randomise_noise:
			fixed_noise = torch.randn(num_images, GEN_LATENT_DIM, 1, 1, device=DEVICE)

		age_bin = slider_age.get()
		age_bin_tensor = torch.full((num_images, 1), age_bin, device=DEVICE, dtype=torch.long)

		gender = stringv_gender.get()
		gender_id = DATASET_DICT['gender'].index(gender)
		gender_id_tensor = torch.full((num_images, 1), gender_id, device=DEVICE, dtype=torch.long)

		race1 = stringv_race1.get()
		race2 = stringv_race2.get()
		race1_id = DATASET_DICT['race'].index(race1)
		race2_id = DATASET_DICT['race'].index(race2)
		race_interp = slider_race.get()
		race_id = race1_id if race_interp == 0 else race2_id
		race_id_tensor = torch.full((num_images, 1), race_id, device=DEVICE, dtype=torch.long)
		race_embedding_interp = None

		labels = torch.cat([age_bin_tensor, gender_id_tensor, race_id_tensor], dim=1)

		if 0 < race_interp < 1:
			race1_embedding = gen_model.race_embedding(torch.tensor([race1_id], device=DEVICE))
			race2_embedding = gen_model.race_embedding(torch.tensor([race2_id], device=DEVICE))
			race_embedding_interp = race_interp * race2_embedding + (1 - race_interp) * race1_embedding
			race_embedding_interp = race_embedding_interp.repeat(num_images, 1)

		with torch.inference_mode():
			fake = gen_model(fixed_noise, labels, race_embedding=race_embedding_interp)

		plt.cla()
		for img, ax in zip(fake, axes.flatten()):
			img = destandardise_transform(img).cpu()
			img = img.permute(1, 2, 0)  # (C, H, W) -> (H, W, C)
			ax.imshow(img)
			ax.axis('off')
		if race_interp == 0:
			title_race = race1
		elif race_interp == 1:
			title_race = race2
		else:
			title_race = f'{race_interp:.2f}({race2}) + {(1 - race_interp):.2f}({race1})'
		plt.suptitle(f'Generator test: age bin #{age_bin}, {gender}, {title_race}', y=0.96, color='white')
		plt.show()

	root = tk.Tk()
	root.title('UTKFace cGAN')
	root.config(width=450, height=300, background='#101010')
	root.resizable(False, False)

	age_lbl = tk.Label(root, text='Select age bin:',
		font='consolas 10', background='#101010', foreground='white')
	age_var = tk.IntVar(value=2)
	age_var.trace_add(mode='write', callback=sample)
	slider_age = tk.Scale(root, from_=0, to=9, resolution=1, variable=age_var, orient='horizontal', font='consolas 10',
		background='#101010', foreground='white', activebackground='#30a0e0', highlightbackground='#101010', borderwidth=0)

	gender_lbl = tk.Label(root, text='Select gender:',
		font='consolas 10', background='#101010', foreground='white')
	stringv_gender = tk.StringVar(value='male')
	stringv_gender.trace_add(mode='write', callback=sample)
	radio_btn_male = tk.Radiobutton(root, text='Male', font='consolas 10', variable=stringv_gender,
		value='male', background='#101010', foreground='white',
		activebackground='#101010', activeforeground='white', selectcolor='#101010')
	radio_btn_female = tk.Radiobutton(root, text='Female', font='consolas 10', variable=stringv_gender,
		value='female', background='#101010', foreground='white',
		activebackground='#101010', activeforeground='white', selectcolor='#101010')

	race_lbl = tk.Label(root, text='Select/interpolate race:',
		font='consolas 10', background='#101010', foreground='white')
	stringv_race1 = tk.StringVar(value='white')
	stringv_race2 = tk.StringVar(value='black')
	stringv_race1.trace_add(mode='write', callback=sample)
	stringv_race2.trace_add(mode='write', callback=sample)
	race_option_menu1 = tk.OptionMenu(root, stringv_race1, *DATASET_DICT['race'])
	race_option_menu1.config(font='consolas 10', indicatoron=0, highlightthickness=0)
	race_option_menu2 = tk.OptionMenu(root, stringv_race2, *DATASET_DICT['race'])
	race_option_menu2.config(font='consolas 10', indicatoron=0, highlightthickness=0)
	race_var = tk.DoubleVar(value=0)
	race_var.trace_add(mode='write', callback=sample)
	slider_race = tk.Scale(root, from_=0, to=1, resolution=0.01, variable=race_var, orient='horizontal', font='consolas 10',
		background='#101010', foreground='white', activebackground='#30a0e0', highlightbackground='#101010', borderwidth=0)

	_, axes = plt.subplots(nrows=4, ncols=6, figsize=(7, 5))
	plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, hspace=0.03, wspace=0.03)
	plt.gcf().set_facecolor('black')

	btn_sample = tk.Button(root, text='Sample images', font='consolas 11', command=lambda: sample(randomise_noise=True))

	age_lbl.place(width=120, height=29, relx=0.5, y=21, anchor='center')
	slider_age.place(width=200, relx=0.5, y=50, anchor='center')
	gender_lbl.place(width=120, height=29, relx=0.5, y=95, anchor='center')
	radio_btn_male.place(width=80, height=29, relx=0.4, y=125, anchor='center')
	radio_btn_female.place(width=80, height=29, relx=0.6, y=125, anchor='center')
	race_lbl.place(width=180, height=29, relx=0.5, y=160, anchor='center')
	race_option_menu1.place(width=80, height=29, relx=0.17, y=191, anchor='center')
	race_option_menu2.place(width=80, height=29, relx=0.83, y=191, anchor='center')
	slider_race.place(width=200, relx=0.5, y=189, anchor='center')
	btn_sample.place(width=150, height=29, relx=0.5, y=260, anchor='center')

	sample()

	root.mainloop()
