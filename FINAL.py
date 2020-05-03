import os
import numpy as np
import argparse
import torch
import time
import librosa
import pickle
import keras
import audio_smoothing
from trainingDataset import trainingDataset
from VC_model import Generator, Discriminator


import os
import time
import argparse
import numpy as np
import pickle

import preprocess


def save_pickle(variable, fileName):
    with open(fileName, 'wb') as f:
        pickle.dump(variable, f)


def load_pickle_file(fileName):
    with open(fileName, 'rb') as f:
        return pickle.load(f)


def preprocess_for_training(train_A_dir, train_B_dir, out_folder):
    num_mcep = 24
    sampling_rate = 16000
    frame_period = 5.0
    n_frames = 128

    start_time = time.time()

    wavs_A = preprocess.load_wavs(wav_dir=train_A_dir, sr=sampling_rate)
    wavs_B = preprocess.load_wavs(wav_dir=train_B_dir, sr=sampling_rate)

    f0s_A, timeaxes_A, sps_A, aps_A, coded_sps_A = preprocess.world_encode_data(
        wave=wavs_A, fs=sampling_rate, frame_period=frame_period, coded_dim=num_mcep)
    f0s_B, timeaxes_B, sps_B, aps_B, coded_sps_B = preprocess.world_encode_data(
        wave=wavs_B, fs=sampling_rate, frame_period=frame_period, coded_dim=num_mcep)

    log_f0s_mean_A, log_f0s_std_A = preprocess.logf0_statistics(f0s=f0s_A)
    log_f0s_mean_B, log_f0s_std_B = preprocess.logf0_statistics(f0s=f0s_B)

    print("Log Pitch in_sample")
    print("Mean: {:.4f}, Std: {:.4f}".format(log_f0s_mean_A, log_f0s_std_A))
    print("Log Pitch out_sample")
    print("Mean: {:.4f}, Std: {:.4f}".format(log_f0s_mean_B, log_f0s_std_B))

    coded_sps_A_transposed = preprocess.transpose_in_list(lst=coded_sps_A)
    coded_sps_B_transposed = preprocess.transpose_in_list(lst=coded_sps_B)

    coded_sps_A_norm, coded_sps_A_mean, coded_sps_A_std = preprocess.coded_sps_normalization_fit_transform(
        coded_sps=coded_sps_A_transposed)
    coded_sps_B_norm, coded_sps_B_mean, coded_sps_B_std = preprocess.coded_sps_normalization_fit_transform(
        coded_sps=coded_sps_B_transposed)

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    np.savez(os.path.join(out_folder, 'logf0s.npz'),
             mean_A=log_f0s_mean_A,
             std_A=log_f0s_std_A,
             mean_B=log_f0s_mean_B,
             std_B=log_f0s_std_B)

    np.savez(os.path.join(out_folder, 'mcep.npz'),
             mean_A=coded_sps_A_mean,
             std_A=coded_sps_A_std,
             mean_B=coded_sps_B_mean,
             std_B=coded_sps_B_std)

    save_pickle(variable=coded_sps_A_norm,
                fileName=os.path.join(out_folder, "in_voice.pickle"))
    save_pickle(variable=coded_sps_B_norm,
                fileName=os.path.join(out_folder, "out_voice.pickle"))

    end_time = time.time()
    print("Preprocessing finsihed . . ! ! !")

    print("Time taken for preprocessing {:.4f} seconds".format(end_time - start_time))


class voice_conversion:
    def __init__(self, logf0s_normalization, mcep_normalization, coded_in, coded_out, model_checkpoint, validation_dir, output_dir):
        self.start_epoch = 0
        self.num_epochs = 5000
        self.mini_batch_size = 1
        self.dataset_A = self.loadPickleFile(coded_in)
        self.dataset_B = self.loadPickleFile(coded_out)
        self.device = 'cpu'

        logf0s_normalization = np.load(logf0s_normalization)
        self.log_f0s_mean_A = logf0s_normalization['mean_A']
        self.log_f0s_std_A = logf0s_normalization['std_A']
        self.log_f0s_mean_B = logf0s_normalization['mean_B']
        self.log_f0s_std_B = logf0s_normalization['std_B']
        mcep_normalization = np.load(mcep_normalization)
        self.coded_sps_A_mean = mcep_normalization['mean_A']
        self.coded_sps_A_std = mcep_normalization['std_A']
        self.coded_sps_B_mean = mcep_normalization['mean_B']
        self.coded_sps_B_std = mcep_normalization['std_B']
        self.generator_A2B = Generator().to(self.device)
        self.generator_B2A = Generator().to(self.device)
        self.discriminator_A = Discriminator().to(self.device)
        self.discriminator_B = Discriminator().to(self.device)
        criterion_mse = torch.nn.MSELoss()
        g_params = list(self.generator_A2B.parameters()) + list(self.generator_B2A.parameters())
        d_params = list(self.discriminator_A.parameters()) + list(self.discriminator_B.parameters())
        self.generator_lr = 0.0002
        self.discriminator_lr = 0.0001
        self.generator_lr_decay = self.generator_lr / 200000
        self.discriminator_lr_decay = self.discriminator_lr / 200000
        self.start_decay = 200000
        self.generator_optimizer = torch.optim.Adam(g_params, lr=self.generator_lr, betas=(0.5, 0.999))
        self.discriminator_optimizer = torch.optim.Adam(d_params, lr=self.discriminator_lr, betas=(0.5, 0.999))
        self.modelCheckpoint = model_checkpoint
        self.validation_dir = validation_dir
        self.output_dir = output_dir
        self.generator_loss_store = []
        self.discriminator_loss_store = []
        self.file_name = 'command_window_store.txt'

    def adjust_lr_rate(self, optimizer, name='generator'):
        if name == 'generator':
            self.generator_lr = max(
                0., self.generator_lr - self.generator_lr_decay)
            for param_groups in optimizer.param_groups:
                param_groups['lr'] = self.generator_lr
        else:
            self.discriminator_lr = max(
                0., self.discriminator_lr - self.discriminator_lr_decay)
            for param_groups in optimizer.param_groups:
                param_groups['lr'] = self.discriminator_lr

    def reset_grad(self):
        self.generator_optimizer.zero_grad()
        self.discriminator_optimizer.zero_grad()

    def train(self):
        for epoch in range(self.start_epoch, self.num_epochs):
            start_time_epoch = time.time()
            cycle_loss_lambda = 10
            identity_loss_lambda = 5
            n_samples = len(self.dataset_A)
            dataset = trainingDataset(datasetA=self.dataset_A,
                                      datasetB=self.dataset_B,
                                      n_frames=128)
            train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                       batch_size=self.mini_batch_size,
                                                       shuffle=True,
                                                       drop_last=False)

            for i, (real_A, real_B) in enumerate(train_loader):

                num_iterations = (
                    n_samples // self.mini_batch_size) * epoch + i
                # print("iteration no: ", num_iterations, epoch)

                if num_iterations > 100:
                    identity_loss_lambda = 0
                if num_iterations > self.start_decay:
                    self.adjust_lr_rate(self.generator_optimizer, name='generator')
                    self.adjust_lr_rate(self.generator_optimizer, name='discriminator')

                real_A = real_A.to(self.device).float()
                real_B = real_B.to(self.device).float()

                fake_B = self.generator_A2B(real_A)
                cycle_A = self.generator_B2A(fake_B)

                fake_A = self.generator_B2A(real_B)
                cycle_B = self.generator_A2B(fake_A)

                identity_A = self.generator_B2A(real_A)
                identity_B = self.generator_A2B(real_B)

                d_fake_A = self.discriminator_A(fake_A)
                d_fake_B = self.discriminator_B(fake_B)

                cycleLoss = torch.mean(
                    torch.abs(real_A - cycle_A)) + torch.mean(torch.abs(real_B - cycle_B))

                identiyLoss = torch.mean(
                    torch.abs(real_A - identity_A)) + torch.mean(torch.abs(real_B - identity_B))

                generator_loss_A2B = torch.mean((1 - d_fake_B) ** 2)
                generator_loss_B2A = torch.mean((1 - d_fake_A) ** 2)

                # Total Generator Loss
                generator_loss = generator_loss_A2B + generator_loss_B2A + \
                    cycle_loss_lambda * cycleLoss + identity_loss_lambda * identiyLoss
                self.generator_loss_store.append(generator_loss.item())

                self.reset_grad()
                generator_loss.backward()

                self.generator_optimizer.step()

                d_real_A = self.discriminator_A(real_A)
                d_real_B = self.discriminator_B(real_B)

                generated_A = self.generator_B2A(real_B)
                d_fake_A = self.discriminator_A(generated_A)

                generated_B = self.generator_A2B(real_A)
                d_fake_B = self.discriminator_B(generated_B)

                d_loss_A_real = torch.mean((1 - d_real_A) ** 2)
                d_loss_A_fake = torch.mean((0 - d_fake_A) ** 2)
                d_loss_A = (d_loss_A_real + d_loss_A_fake) / 2.0

                d_loss_B_real = torch.mean((1 - d_real_B) ** 2)
                d_loss_B_fake = torch.mean((0 - d_fake_B) ** 2)
                d_loss_B = (d_loss_B_real + d_loss_B_fake) / 2.0

                d_loss = (d_loss_A + d_loss_B) / 2.0
                self.discriminator_loss_store.append(d_loss.item())

                self.reset_grad()
                d_loss.backward()

                self.discriminator_optimizer.step()
                if num_iterations % 50 == 0:
                    store_to_file = "Iter:{}\t Generator Loss:{:.4f} Discrimator Loss:{:.4f}".format(
                        num_iterations, generator_loss.item(), d_loss.item())
                    print("Iter:{}\t Generator Loss:{:.4f} Discrimator Loss:{:.4f} ".format(
                        num_iterations, generator_loss.item(), d_loss.item()))
                    self.store_to_file(store_to_file)
            end_time = time.time()
            store_to_file = "Epoch: {} Generator Loss: {:.4f} Discriminator Loss: {}, Time: {:.2f}\n\n".format(
                epoch, generator_loss.item(), d_loss.item(), end_time - start_time_epoch)
            self.store_to_file(store_to_file)
            print("Epoch: {} , Time taken : {:.2f}\n".format(epoch, end_time - start_time_epoch))

            if epoch != 0:
                print("Saving model Checkpoint  ......")
                store_to_file = "Saving model Checkpoint  ......"
                self.store_to_file(store_to_file)
                self.saveModelCheckPoint(epoch, '{}'.format( self.modelCheckpoint + '_CycleGAN_CheckPoint'))
                print("Model Saved!")

            if epoch != 0:
                print('validating . . . ! ! !')
                validation_start_time = time.time()
                self.validation_for_A_dir()
                validation_end_time = time.time()
                store_to_file = "Time taken for validation Set: {}".format(
                    validation_end_time - validation_start_time)
                self.store_to_file(store_to_file)
                print("Time taken for validation Set: {}".format(
                    validation_end_time - validation_start_time))

    def validation_for_A_dir(self):
        num_mcep = 24
        sampling_rate = 16000
        frame_period = 5.0
        n_frames = 128
        validation_dir = self.validation_dir
        output_dir = self.output_dir

        for file in os.listdir(validation_dir):
            filePath = os.path.join(validation_dir, file)
            wav, _ = librosa.load(filePath, sr=sampling_rate, mono=True)
            wav = audio_smoothing.wav_padding(wav=wav,
                                         sr=sampling_rate,
                                         frame_period=frame_period,
                                         multiple=4)
            f0, timeaxis, sp, ap = audio_smoothing.world_decompose(
                wav=wav, fs=sampling_rate, frame_period=frame_period)
            f0_converted = audio_smoothing.pitch_conversion(f0=f0,
                                                       mean_log_src=self.log_f0s_mean_A,
                                                       std_log_src=self.log_f0s_std_A,
                                                       mean_log_target=self.log_f0s_mean_B,
                                                       std_log_target=self.log_f0s_std_B)
            coded_sp = audio_smoothing.world_encode_spectral_envelop(
                sp=sp, fs=sampling_rate, dim=num_mcep)
            coded_sp_transposed = coded_sp.T
            coded_sp_norm = (coded_sp_transposed -
                             self.coded_sps_A_mean) / self.coded_sps_A_std
            coded_sp_norm = np.array([coded_sp_norm])

            if torch.cuda.is_available():
                coded_sp_norm = torch.from_numpy(coded_sp_norm).cuda().float()
            else:
                coded_sp_norm = torch.from_numpy(coded_sp_norm).float()

            coded_sp_converted_norm = self.generator_A2B(coded_sp_norm)
            coded_sp_converted_norm = coded_sp_converted_norm.cpu().detach().numpy()
            coded_sp_converted_norm = np.squeeze(coded_sp_converted_norm)
            coded_sp_converted = coded_sp_converted_norm * \
                self.coded_sps_B_std + self.coded_sps_B_mean
            coded_sp_converted = coded_sp_converted.T
            coded_sp_converted = np.ascontiguousarray(coded_sp_converted)
            decoded_sp_converted = audio_smoothing.world_decode_spectral_envelop(
                coded_sp=coded_sp_converted, fs=sampling_rate)
            wav_transformed = audio_smoothing.world_speech_synthesis(f0=f0_converted,
                                                                decoded_sp=decoded_sp_converted,
                                                                ap=ap,
                                                                fs=sampling_rate,
                                                                frame_period=frame_period)
            librosa.output.write_wav(path=os.path.join(output_dir, os.path.basename(file)),
                                     y=wav_transformed,
                                     sr=sampling_rate)

    def savePickle(self, variable, fileName):
        with open(fileName, 'wb') as f:
            pickle.dump(variable, f)

    def loadPickleFile(self, fileName):
        with open(fileName, 'rb') as f:
            return pickle.load(f)

    def store_to_file(self, doc):
        doc = doc + "\n"
        with open(self.file_name, "a") as myfile:
            myfile.write(doc)

    def saveModelCheckPoint(self, epoch, PATH):
        torch.save({
            'epoch': epoch,
            'generator_loss_store': self.generator_loss_store,
            'discriminator_loss_store': self.discriminator_loss_store,
            'model_genA2B_state_dict': self.generator_A2B.state_dict(),
            'model_genB2A_state_dict': self.generator_B2A.state_dict(),
            'model_discriminatorA': self.discriminator_A.state_dict(),
            'model_discriminatorB': self.discriminator_B.state_dict(),
            'generator_optimizer': self.generator_optimizer.state_dict(),
            'discriminator_optimizer': self.discriminator_optimizer.state_dict()
        }, PATH)

    def loadModel(self, PATH):
        checkPoint = torch.load(PATH)
        self.generator_A2B.load_state_dict(
            state_dict=checkPoint['model_genA2B_state_dict'])
        self.generator_B2A.load_state_dict(
            state_dict=checkPoint['model_genB2A_state_dict'])
        self.discriminator_A.load_state_dict(
            state_dict=checkPoint['model_discriminatorA'])
        self.discriminator_B.load_state_dict(
            state_dict=checkPoint['model_discriminatorB'])
        self.generator_optimizer.load_state_dict(
            state_dict=checkPoint['generator_optimizer'])
        self.discriminator_optimizer.load_state_dict(
            state_dict=checkPoint['discriminator_optimizer'])
        epoch = int(checkPoint['epoch']) + 1
        self.generator_loss_store = checkPoint['generator_loss_store']
        self.discriminator_loss_store = checkPoint['discriminator_loss_store']
        return epoch






train_A_dir = './data/in_samples/'
train_B_dir = './data/out_samples/'
out_folder = './data/'

print()

preprocess_for_training(train_A_dir, train_B_dir, out_folder)

print('__________________________________________________')
print(' ')

logf0s_normalization = './data/logf0s.npz'
mcep_normalization = './data/mcep.npz'
coded_in = './data/in_voice.pickle'
coded_out = './data/out_voice.pickle'
model_checkpoint = './data/'
resume_training_at = None

validation_dir = './data/in_samples/'
output_dir = './data/output/'

cycleGAN = voice_conversion(logf0s_normalization,mcep_normalization,coded_in,coded_out,model_checkpoint, validation_dir, output_dir)
cycleGAN.train()

