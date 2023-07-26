import os
import glob
import yaml
import torch
import random
import pickle as pk
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_video
from torch.utils.data import DataLoader


class m1_mydataset(Dataset):
    def __init__(self, dilation, f_size, prediction_span, dataset_name, mode, checkpoint_path, anchor=None):
        super().__init__()
        self.dataset = dataset_name
        self.f_size = f_size # frame size: the number of frames you want to look backback
        self.dilation = dilation # distance between two anchors while downsampling along time axis
        self.ps = prediction_span # how many frames/timesteps later the model will predict/look forward. ps=1 -> next frame/timestep, ps=2 -> the frame after next frame
        self.mode = mode
        self.splits = [0.8, 0.1, 0.1]
        if self.dataset == 'vir_poppy':
            assert anchor, 'Virtual Poppy dataset needs a parameter named anchor for cutting the head (of videos) off when extracting video clips'
        else:
            assert not anchor, 'Real Poppy dataset does not require parameter anchor'
        
        # what dataset you use
        if self.dataset == 'vir_poppy':
            self.anchor = anchor
            video_list = {'line': []}
            if os.path.exists(os.path.join(os.getcwd(), 'virtual_poppy', 'videos.h5')):
                video_list = pd.read_hdf(os.path.join(os.getcwd(), 'virtual_poppy', 'videos.h5'))
                if not os.path.exists(os.path.join(checkpoint_path, 'rep.pkl')):
                    # repetitions
                    rep = self.split(video_list)
                    with open(os.path.join(checkpoint_path, 'rep.pkl'), 'wb') as file:
                        pk.dump(rep, file)       
                else:
                    with open(os.path.join(checkpoint_path, 'rep.pkl'), 'rb') as f:
                        rep = pk.load(f)
            else:
                raise Exception('You should run utils.preprocessing for the virtual Poppy dataset before initiate a dataset.')
            
            # training/validation/testing/example testing
            if self.mode == 'training':
                idx = rep[0]
            elif self.mode == 'validation':
                idx = rep[1]
            elif self.mode == 'testing':
                idx = rep[2]
            elif self.mode == 'example':
                # default: 2 examples
                idx = random.sample(rep[2], 2)

            video_list = video_list.loc[idx]
            self.frames_labels_idx = self.get_lables_frames_VP(video_list)
            print(f'{self.dataset} {self.mode} dataset: {len(self.frames_labels_idx)} video clips from {len(video_list)} videos')
        elif self.dataset == 'real_poppy':
            path = os.path.join(os.getcwd(), 'real_poppy')
            num_reps = glob.glob(os.path.join(path, 'frames_-*.pkl'))
            assert len(num_reps) == len(glob.glob(os.path.join(path, 'results_-*.pkl'))), 'Incomplete dataset!'
            self.splits = [int(len(num_reps)*i) for i in self.splits]
            num_reps = list(range(len(num_reps)))
            if self.mode == 'training':
                num_reps = random.sample(num_reps, self.splits[0])
            elif self.mode == 'validation':
                sample = random.sample(num_reps, self.splits[0])
                num_reps = [i for i in num_reps if i not in sample]
                num_reps = random.sample(num_reps, self.splits[1])
            elif self.mode == 'testing' or self.mode == 'example':
                sample = random.sample(num_reps, self.splits[0])
                num_reps = [i for i in num_reps if i not in sample]
                sample = random.sample(num_reps, self.splits[1])
                num_reps = [i for i in num_reps if i not in sample]
                num_reps = random.sample(num_reps, self.splits[2])
            print(f'{self.mode}: {[ele + 1 for ele in num_reps]}')
            '''
            results[rep]: the number of successful transitions before a fall (4 is a complete step without falling)
            bufs[rep][k][i]: buffer data for ith waypoint of kth trajectory in repetition rep
                bufs[rep][k][i] is a tuple (flag, buffers, elapsed)
                flag: True if motion was completed safely (e.g., low motor temperature), False otherwise
                buffers['position'][t, j]: actual angle of jth joint at timestep t of motion
                buffers['target'][t, j]: target angle of jth joint at timestep t of motion
                elapsed[t]: elapsed time since start of motion at timestep t of motion
            ''' 
            # load all the data
            results, bufs, frames = {}, {}, {}
            for rep in num_reps:
                with open(os.path.join(path, 'results_%d.pkl' % (-rep-1)), 'rb') as f:
                    (_, results[rep], bufs[rep]) = pk.load(f, encoding='latin1') # motor_names, results[rep], bufs[rep]

                with open(os.path.join(path, 'frames_%d.pkl' % (-rep-1)), 'rb') as f:
                    frames[rep] = pk.load(f, encoding='latin1')
            
            self.frames_labels_idx = self.get_labels_frames_RP(num_reps, results, bufs, frames)

    def __len__(self):
        return len(self.frames_labels_idx)
    
    def __getitem__(self, index):
        if self.mode == 'example':
            return self.frames_labels_idx[index][0], self.frames_labels_idx[index][1], self.frames_labels_idx[index][2]
        else:
            video_clip = torch.tensor(self.frames_labels_idx[index][0]).float().permute(3,0,1,2) / 255 # tensor(C, f_size, H, W)
            idx = self.frames_labels_idx[index][1]
            return video_clip, idx
    
    def split(self, video_list):
        half_length = int(len(video_list['folder'])/2)

        # split training, validation and testing set for each fold
        splits = [int(half_length*i) for i in self.splits]
        rep = []
    
        remained_idx = list(range(half_length))
        # training
        training = random.sample(remained_idx, splits[0])
        remained_idx = list(set(remained_idx).difference(training))
        training = training+[half_length+i for i in training]
        random.shuffle(training)
        # validation
        validation = random.sample(remained_idx, splits[1])
        remained_idx = list(set(remained_idx).difference(validation))
        validation = validation+[half_length+i for i in validation]
        random.shuffle(validation)
        # testing
        testing = random.sample(remained_idx, splits[2])
        remained_idx = list(set(remained_idx).difference(testing))
        testing = testing+[half_length+i for i in testing]
        random.shuffle(testing)

        rep.append(training)
        rep.append(validation)
        rep.append(testing)
        return rep
    
    def get_lables_frames_VP(self, video_list):
        frames_lables_idx = []
        
        for idx, (folder, clss, line) in enumerate(zip(list(video_list['folder']), list(video_list['class']), list(video_list['line']))):
            
            path = os.path.join(os.getcwd(), 'virtual_poppy', clss, folder)
            sp_anchor = len(glob.glob(os.path.join(path, '*.png')))-1 # stop anchor is the highest index of images of this full video
            frames = []
            indexes = []
            anchor = self.anchor
            while anchor <= sp_anchor:
                img = Image.open(os.path.join(path, '{}.png'.format(anchor)))
                img.load()
                img = np.asarray(img.convert('RGB'), dtype='uint8')
                frames.append(img)
                indexes.append(anchor)
                anchor += self.dilation

            # with open(os.path.join(path, 'hc_z.pkl'), 'rb') as f:
            #     hc_z_coordinates = [coordinate[2] for coordinate in pk.load(f)]
            #     hc_z_coordinates = np.array(hc_z_coordinates)[indexes]

            # 1 -> fall, 0 -> standing
            # label = [None] + list((np.absolute(hc_z_coordinates[1:] - hc_z_coordinates[:-1])>0.00169).astype(int))
            # line = self.set_line(label)
            # length = len(frames_lables_idx)
            
            if self.mode == 'example':
                frames_lables_idx.append((np.stack(frames[line - self.f_size : line]), line - self.f_size, path.split(os.sep)[-1]))
            else:
                anchor = random.randint(self.f_size, line) # anchor could be arbitrary timestep as long as it is in the range: (self.f_size, len(frames))
                frames_lables_idx.append((np.stack(frames[anchor - self.f_size : anchor]), anchor)) # autoencoder does not need prediction span for indexing or extracting vc

            # print(f'Get {len(frames_lables_idx) - length} video clips, terminating line: {line}, {idx + 1}/{len(video_list)}')

        return frames_lables_idx
    
    def get_labels_frames_RP(self, num_reps, results, bufs, frames):

        frames_lables_idx = []
        for count, rep in enumerate(num_reps):
                 
            elapsed_rep = []
            frames_temp = []
            actuals = []
            
            '''
            Near the end there is a transition to lift, and then a transition to kick.
            In practice, these two transitions happened very quickly, because one of Poppy's feet does not touch the ground,
            and it can easily lose balance if it stays on only one foot for too long.
            They were in fact so quick that I could not reliably distinguish them when I recorded successes by hand.
            So these two transitions are lumped together for the purposes of recording success.
            '''
            if results[rep] >= 3:
                checkpoint = results[rep] + 1
                line = (22 + 2 + 3 * (results[rep] - 2)) * 10 - self.ps
            else:
                checkpoint = results[rep]
                line = (11 * results[rep]) * 10 - self.ps
            if line == 0:
                print(f'line: {line}')
                print(f'{len(frames_lables_idx)} from this repetition')
                continue

            for k in range(5):
                _, buffers, elapsed = zip(*bufs[rep][k]) # (flag, buffers, elapsed)
                
                # actual labels
                if k < checkpoint:
                    label = 0
                else:
                    label = 1
                for _ in range(len(buffers)*10): # 10 waypoints/move
                    actuals.append(label)
                # accumulate elapsed time over multiple waypoints
                for i in range(1, len(elapsed)):
                    elapsed[i][:] = elapsed[i] + elapsed[i-1][-1]
                elapsed = np.concatenate(elapsed)
                elapsed_rep.append(elapsed)
                # video clip
                frames_temp.append(np.concatenate(np.stack(frames[rep][k])))

            actuals = np.array(actuals)
            for i in range(1, len(elapsed_rep)):
                elapsed_rep[i][:] = elapsed_rep[i] + elapsed_rep[i-1][-1]
            elapsed_rep = np.concatenate(elapsed_rep)
            frames_temp = np.concatenate(frames_temp)

            # linear interpolation: frames and positions
            interval = elapsed_rep[-1]/elapsed_rep.shape[0]
            destination_anchor = int(elapsed_rep.shape[0]/self.dilation)
            idx = [interval * (i+1) * self.dilation for i in range(destination_anchor)]
            upper_bound_idx = np.searchsorted(elapsed_rep, idx)
            lower_bound_idx = upper_bound_idx -1

            intpolat_frames, intpolat_actuals = [], []
            # reason for -1: upper_bound_idx[-1] is len(elapsed_rep), which will results in "index exceeds" error
            for i in range(destination_anchor-1):  
                # [frames[lower] * (distance btwn idx[i] and its upper bound) + frames[upper] * (distance btwn idx[i] and its lower bound)]
                new_frame = frames_temp[lower_bound_idx[i]]*(elapsed_rep[upper_bound_idx[i]] - idx[i]) + frames_temp[upper_bound_idx[i]]*(idx[i]-elapsed_rep[lower_bound_idx[i]])
                new_frame /= elapsed_rep[upper_bound_idx[i]] - elapsed_rep[lower_bound_idx[i]]
                intpolat_frames.append(new_frame)
                
                # # [actuals[lower] * (distance btwn idx[i] and its upper bound) + actuals[upper] * (distance btwn idx[i] and its lower bound)]
                # intpolat_actuals.append(actuals[lower_bound_idx[i]]*(elapsed_rep[upper_bound_idx[i]] - idx[i]) + 
                #                         actuals[upper_bound_idx[i]]*(idx[i]-elapsed_rep[lower_bound_idx[i]]))

                # actuals[lower_bound_idx[i]] or actuals[higher_bound_idx[i]], which one is closer to idx[i]
                if elapsed_rep[upper_bound_idx[i]] - idx[i] >= idx[i]-elapsed_rep[lower_bound_idx[i]]:
                    new_actuals = actuals[lower_bound_idx[i]]
                else:
                    new_actuals = actuals[upper_bound_idx[i]]
                intpolat_actuals.append(new_actuals)
                
            # append the last image (frames_temp[-1]) to make up the dropped image
            # Note: frames_temp[-1] is not always frames[rep][-1][-1][-1] since the results[rep] could be smaller than 4
            intpolat_frames.append(frames_temp[-1])

            # append the last position (actuals[-1]) to make up the dropped image
            intpolat_actuals.append(actuals[-1])

            length = len(frames_lables_idx)
            if self.mode == 'example':
                anchor = random.randint(self.f_size, line)
                frames_lables_idx.append((np.stack(intpolat_frames[anchor - self.f_size : anchor]), anchor + self.ps - 1, f'frames_{-rep-1}.pkl'))

            else:
                anchor = self.f_size
                while anchor + self.ps <= line:
                    frames_lables_idx.append((np.stack(intpolat_frames[anchor - self.f_size : anchor]), anchor - 1))
                    anchor += 10

            print(f'Get {len(frames_lables_idx) - length} video clips, line: {line}, {count + 1}/{len(num_reps)}')

        return frames_lables_idx
    
    def tell_me_HW(self):
        return self.frames_labels_idx[0][0].shape[:3]

class m2_mydataset(Dataset):
    def __init__(self, checkpoint_path, seed):
        super().__init__()
        with open(os.path.join(checkpoint_path, 'HyperParam.yml'), 'r') as file:
            HpParams = yaml.safe_load(file)
        self.f_size = HpParams['f_size'] # frame size: the number of frames you want to look backback
        self.dilation = HpParams['dilation'] # distance between two anchors while downsampling along time axis
        self.ps = HpParams['prediction_span'] # how many frames/timesteps later the model will predict/look forward. ps=1 -> next frame/timestep, ps=2 -> the frame after next frame
        dataset_name = HpParams['dataset_name']
        if dataset_name == 'vir_poppy':
            self.anchor = HpParams['anchor']
            video_list = pd.read_hdf(os.path.join(os.getcwd(), 'virtual_poppy', 'videos.h5'))
            qualified = video_list[5000:][video_list.iloc[5000:]['line'] >= self.f_size + self.ps - 1]
            chosen = pd.concat([video_list.iloc[:5000].sample(n= int(0.8*len(qualified)), random_state= seed), qualified.sample(n= int(0.8*len(qualified)), random_state= seed)])
            for_sup = pd.concat([video_list, chosen]).drop_duplicates(keep=False)
            for_sup.to_hdf(os.path.join(checkpoint_path, str(seed), 'sup.h5'), key='sup')
            self.frames_labels_idx = self.get_lables_frames_VP(chosen)
            print(f'{dataset_name} dataset: {len(self.frames_labels_idx)} video clips from {len(chosen)} videos')
        elif dataset_name == 'real_poppy':
            path = os.path.join(os.getcwd(), 'real_poppy')
            num_reps = glob.glob(os.path.join(path, 'frames_-*.pkl'))
            assert len(num_reps) == len(glob.glob(os.path.join(path, 'results_-*.pkl'))), 'Incomplete dataset!'
            num_reps = list(range(len(num_reps)))
            '''
            results[rep]: the number of successful transitions before a fall (4 is a complete step without falling)
            bufs[rep][k][i]: buffer data for ith waypoint of kth trajectory in repetition rep
                bufs[rep][k][i] is a tuple (flag, buffers, elapsed)
                flag: True if motion was completed safely (e.g., low motor temperature), False otherwise
                buffers['position'][t, j]: actual angle of jth joint at timestep t of motion
                buffers['target'][t, j]: target angle of jth joint at timestep t of motion
                elapsed[t]: elapsed time since start of motion at timestep t of motion
            '''
            # load all data
            results, bufs, frames = {}, {}, {}
            for rep in num_reps:
                with open(os.path.join(path, 'results_%d.pkl' % (-rep-1)), 'rb') as f:
                    (_, results[rep], bufs[rep]) = pk.load(f, encoding='latin1') # motor_names, results[rep], bufs[rep]

                with open(os.path.join(path, 'frames_%d.pkl' % (-rep-1)), 'rb') as f:
                    frames[rep] = pk.load(f, encoding='latin1')
            tem_df = pd.DataFrame({'results': list(results.values())})
            for_sup = list(tem_df[tem_df['results'] < 4].sample(5, random_state=seed).index.values)
            with open(os.path.join(checkpoint_path, str(seed), 'sup.pkl'), 'wb') as file:
                pk.dump(for_sup, file)
            for idx in for_sup:
                results.pop(idx)
                bufs.pop(idx)
                frames.pop(idx)
                num_reps.remove(idx)
            self.frames_labels_idx = self.get_labels_frames_RP(num_reps, results, bufs, frames)
            print(f'ps: {self.ps}, {dataset_name} dataset: {len(self.frames_labels_idx)} video clips from {len(num_reps)} videos')

    def get_labels_frames_RP(self, num_reps, results, bufs, frames):

        frames_lables_idx = []
        for count, rep in enumerate(num_reps):
            elapsed_rep = []
            frames_temp = []
            actuals = []
            
            '''
            Near the end there is a transition to lift, and then a transition to kick.
            In practice, these two transitions happened very quickly, because one of Poppy's feet does not touch the ground,
            and it can easily lose balance if it stays on only one foot for too long.
            They were in fact so quick that I could not reliably distinguish them when I recorded successes by hand.
            So these two transitions are lumped together for the purposes of recording success.
            '''
            if results[rep] >= 3:
                checkpoint = results[rep] + 1
                line = (22 + 2 + 3 * (results[rep] - 2)) * 10 - self.ps
            else:
                checkpoint = results[rep]
                line = (11 * results[rep]) * 10 - self.ps

            for k in range(5):
                _, buffers, elapsed = zip(*bufs[rep][k]) # (flag, buffers, elapsed)
                
                # actual labels
                if k < checkpoint:
                    label = 0 # standing/walking
                else:
                    label = 1 # fall
                for _ in range(len(buffers)*10): # 10 waypoints/move
                    actuals.append(label)
                # accumulate elapsed time over multiple waypoints
                for i in range(1, len(elapsed)):
                    elapsed[i][:] = elapsed[i] + elapsed[i-1][-1]
                elapsed = np.concatenate(elapsed)
                elapsed_rep.append(elapsed)
                # video clip
                frames_temp.append(np.concatenate(np.stack(frames[rep][k])))

            actuals = np.array(actuals)
            for i in range(1, len(elapsed_rep)):
                elapsed_rep[i][:] = elapsed_rep[i] + elapsed_rep[i-1][-1]
            elapsed_rep = np.concatenate(elapsed_rep)
            frames_temp = np.concatenate(frames_temp)

            # linear interpolation: frames and positions
            interval = elapsed_rep[-1]/elapsed_rep.shape[0]
            destination_anchor = int(elapsed_rep.shape[0]/self.dilation)
            idx = [interval * (i+1) * self.dilation for i in range(destination_anchor)]
            upper_bound_idx = np.searchsorted(elapsed_rep, idx)
            lower_bound_idx = upper_bound_idx -1

            intpolat_frames, intpolat_actuals = [], []
            # reason for -1: upper_bound_idx[-1] is len(elapsed_rep), which will results in "index exceeds" error
            for i in range(destination_anchor-1):  
                # [frames[lower] * (distance btwn idx[i] and its upper bound) + frames[upper] * (distance btwn idx[i] and its lower bound)]
                new_frame = frames_temp[lower_bound_idx[i]]*(elapsed_rep[upper_bound_idx[i]] - idx[i]) + frames_temp[upper_bound_idx[i]]*(idx[i]-elapsed_rep[lower_bound_idx[i]])
                new_frame /= elapsed_rep[upper_bound_idx[i]] - elapsed_rep[lower_bound_idx[i]]
                intpolat_frames.append(new_frame)
                
                # # [actuals[lower] * (distance btwn idx[i] and its upper bound) + actuals[upper] * (distance btwn idx[i] and its lower bound)]
                # intpolat_actuals.append(actuals[lower_bound_idx[i]]*(elapsed_rep[upper_bound_idx[i]] - idx[i]) + 
                #                         actuals[upper_bound_idx[i]]*(idx[i]-elapsed_rep[lower_bound_idx[i]]))

                # actuals[lower_bound_idx[i]] or actuals[higher_bound_idx[i]], which one is closer to idx[i]
                if elapsed_rep[upper_bound_idx[i]] - idx[i] >= idx[i]-elapsed_rep[lower_bound_idx[i]]:
                    new_actuals = actuals[lower_bound_idx[i]]
                else:
                    new_actuals = actuals[upper_bound_idx[i]]
                intpolat_actuals.append(new_actuals)
                
            # append the last image (frames_temp[-1]) to make up the dropped image
            # Note: frames_temp[-1] is not always frames[rep][-1][-1][-1] since the results[rep] could be smaller than 4
            intpolat_frames.append(frames_temp[-1])

            # append the last position (actuals[-1]) to make up the dropped image
            intpolat_actuals.append(actuals[-1])
            
            if results[rep] == 4: # a non-fall episode
                anchor = random.randint(self.f_size, line)
                frames_lables_idx.append((np.stack(intpolat_frames[anchor - self.f_size : anchor]), intpolat_actuals[anchor + self.ps - 1], anchor + self.ps - 1))
            else:
                frames_lables_idx.append((np.stack(intpolat_frames[line - self.f_size : line]), intpolat_actuals[line + self.ps - 1], line + self.ps - 1))

        return frames_lables_idx

    def get_lables_frames_VP(self, video_list):
        frames_lables_idx = []
        
        for idx, (folder, clss, line) in enumerate(zip(list(video_list['folder']), list(video_list['class']), list(video_list['line']))):
            
            path = os.path.join(os.getcwd(), 'virtual_poppy', clss, folder)
            sp_anchor = len(glob.glob(os.path.join(path, '*.png')))-1 # stop anchor is the highest index of images of this full video
            frames = []
            indexes = []
            shifted_indexes = []
            anchor = self.anchor
            while anchor <= sp_anchor:
                img = Image.open(os.path.join(path, '{}.png'.format(anchor)))
                img.load()
                img = np.asarray(img.convert('RGB'), dtype='uint8')
                frames.append(img)
                indexes.append(anchor)
                shifted_indexes.append(anchor - self.dilation)
                anchor += self.dilation

            with open(os.path.join(path, 'hc_z.pkl'), 'rb') as f:
                hc_z_coordinates = [coordinate[2] for coordinate in pk.load(f)]
                hc_z_coordinates_shifted = np.array(hc_z_coordinates)[shifted_indexes]
                hc_z_coordinates = np.array(hc_z_coordinates)[indexes]

            # 1 -> fall, 0 -> standing
            label = list((np.absolute(hc_z_coordinates - hc_z_coordinates_shifted)>0.00169).astype(int))
            if clss == 'standing': # a non-fall episode
                # anchor = random.randint(self.f_size, line - self.ps + 1)
                # frames_lables_idx.append((np.stack(frames[anchor - self.f_size : anchor]), label[anchor + self.ps - 1], anchor + self.ps - 1))
                pass
            elif clss == 'fall_new':
                frames_lables_idx.append((np.stack(frames[line - self.ps - self.f_size + 1: line - self.ps + 1]), label[line], line)) # vc labeled as "FALL"
                line_StandingInFall = [i + self.f_size + self.ps - 1 for i in range(len(label[self.f_size + self.ps - 1: line])) if label[self.f_size + self.ps - 1: line][i] == 0]
                if len(line_StandingInFall) > 0:
                    line_SIF = random.choice(line_StandingInFall)
                    frames_lables_idx.append((np.stack(frames[line_SIF - self.ps - self.f_size + 1: line_SIF - self.ps + 1]), label[line_SIF], line_SIF)) # vc labeled as "STANDING"
        return frames_lables_idx
  
    def __len__(self):
        return len(self.frames_labels_idx)
    
    def __getitem__(self, index):
        video_clip_curr = torch.tensor(self.frames_labels_idx[index][0]).float().permute(3,0,1,2) / 255 # tensor(C, f_size, H, W)
        label = torch.tensor(self.frames_labels_idx[index][1]).float() # tensor(f_size)
        idx = self.frames_labels_idx[index][2]
        return video_clip_curr, label, idx
    
    def tell_me_HW(self):
        return self.frames_labels_idx[0][0].shape[:3]
            
class baseline_dataset(Dataset):
    def __init__(self, dataset_name):
        super().__init__()
        if dataset_name == 'vir_poppy':
            path = os.path.join(os.getcwd(), 'virtual_poppy')
            video_list = pd.read_hdf(os.path.join(os.getcwd(), 'virtual_poppy', 'videos.h5'))
            self.frames_labels = self.get_labels_frames_VP(video_list)
            print(f'{dataset_name} dataset: {len(self.frames_labels)} video clips from {len(video_list)} videos')
        elif dataset_name == 'real_poppy':
            path = os.path.join(os.getcwd(), 'real_poppy')
            num_reps = glob.glob(os.path.join(path, 'frames_-*.pkl'))
            assert len(num_reps) == len(glob.glob(os.path.join(path, 'results_-*.pkl'))), 'Incomplete dataset!'
            num_reps = list(range(len(num_reps)))
            '''
            results[rep]: the number of successful transitions before a fall (4 is a complete step without falling)
            bufs[rep][k][i]: buffer data for ith waypoint of kth trajectory in repetition rep
                bufs[rep][k][i] is a tuple (flag, buffers, elapsed)
                flag: True if motion was completed safely (e.g., low motor temperature), False otherwise
                buffers['position'][t, j]: actual angle of jth joint at timestep t of motion
                buffers['target'][t, j]: target angle of jth joint at timestep t of motion
                elapsed[t]: elapsed time since start of motion at timestep t of motion
            '''
            # load all data
            results, bufs, frames = {}, {}, {}
            for rep in num_reps:
                with open(os.path.join(path, 'results_%d.pkl' % (-rep-1)), 'rb') as f:
                    (_, results[rep], bufs[rep]) = pk.load(f, encoding='latin1') # motor_names, results[rep], bufs[rep]

                with open(os.path.join(path, 'frames_%d.pkl' % (-rep-1)), 'rb') as f:
                    frames[rep] = pk.load(f, encoding='latin1')

            self.frames_labels = self.get_labels_frames_RP(num_reps, results, bufs, frames)
            print(f'{dataset_name} dataset: {len(self.frames_labels)} video clips from {len(num_reps)} videos')
        else:
            raise Exception('Check your spelling, please. real_poppy or vir_poppy?')
    
    def __len__(self):
        return len(self.frames_labels)
    
    def __getitem__(self, index):
        video_clip = torch.tensor(self.frames_labels[index][0]).float().permute(0,3,1,2) / 255 # tensor(f_size, C, H, W)
        label = torch.tensor(self.frames_labels[index][1]).float() # tensor(f_size, )
        return video_clip, label

    def tell_me_HW(self):
        return self.frames_labels[0][0].shape[1:3]
    
    def get_labels_frames_VP(self, video_list):
        frames_lables_idx = []
        for folder, clss in zip(list(video_list['folder']), list(video_list['class'])):
            path = os.path.join(os.getcwd(), 'virtual_poppy', clss, folder)
            sp_anchor = len(glob.glob(os.path.join(path, '*.png'))) - 1 # stop anchor is the highest index of images of this full video
            frames = []
            indexes = [50 + int(len(range(50, sp_anchor + 1)) / 10) * t for t in range(10)]
            for i in indexes:
                img = Image.open(os.path.join(path, '{}.png'.format(i)))
                img.load()
                img = np.asarray(img.convert('RGB'), dtype='uint8')
                frames.append(img)

            # 1 -> fall, 0 -> standing
            if clss == 'standing': # a non-fall episode
                frames_lables_idx.append((np.array(frames), 0))
            elif clss == 'fall_new':
                frames_lables_idx.append((np.array(frames), 1))

        return frames_lables_idx        

    def get_labels_frames_RP(self, num_reps, results, bufs, frames):
        frames_lables_idx = []
        for rep in num_reps:
            elapsed_rep = []
            frames_temp = []
            for k in range(5):
                _, _, elapsed = zip(*bufs[rep][k]) # (flag, buffers, elapsed)
                # accumulate elapsed time over multiple waypoints
                for i in range(1, len(elapsed)):
                    elapsed[i][:] = elapsed[i] + elapsed[i-1][-1]
                elapsed = np.concatenate(elapsed)
                elapsed_rep.append(elapsed)
                # video clip
                frames_temp.append(np.concatenate(np.stack(frames[rep][k])))

            for i in range(1, len(elapsed_rep)):
                elapsed_rep[i][:] = elapsed_rep[i] + elapsed_rep[i-1][-1]
            elapsed_rep = np.concatenate(elapsed_rep)
            frames_temp = np.concatenate(frames_temp)

            # linear interpolation: frames and positions
            interval = elapsed_rep[-1]/elapsed_rep.shape[0]
            destination_anchor = int(elapsed_rep.shape[0]/1)
            idx = [interval * (i+1) * 1 for i in range(destination_anchor)]
            upper_bound_idx = np.searchsorted(elapsed_rep, idx)
            lower_bound_idx = upper_bound_idx -1

            intpolat_frames = []
            # reason for -1: upper_bound_idx[-1] is len(elapsed_rep), which will results in "index exceeds" error
            for i in range(destination_anchor-1):  
                # [frames[lower] * (distance btwn idx[i] and its upper bound) + frames[upper] * (distance btwn idx[i] and its lower bound)]
                new_frame = frames_temp[lower_bound_idx[i]]*(elapsed_rep[upper_bound_idx[i]] - idx[i]) + frames_temp[upper_bound_idx[i]]*(idx[i]-elapsed_rep[lower_bound_idx[i]])
                new_frame /= elapsed_rep[upper_bound_idx[i]] - elapsed_rep[lower_bound_idx[i]]
                intpolat_frames.append(new_frame)

            # append the last image (frames_temp[-1]) to make up the dropped image
            # Note: frames_temp[-1] is not always frames[rep][-1][-1][-1] since the results[rep] could be smaller than 4
            intpolat_frames.append(frames_temp[-1])
            
            chosen_frames = []
            indexes = [50 + int(len(range(50, len(intpolat_frames))) / 10) * t for t in range(10)]
            for i in indexes:
                chosen_frames.append(intpolat_frames[i])

            # 1 -> fall, 0 -> standing
            if results[rep] == 4: # a non-fall episode
                frames_lables_idx.append((np.array(chosen_frames), 0))
            else:
                frames_lables_idx.append((np.array(chosen_frames), 1))

        return frames_lables_idx
    
class m2_mydataset_sup(Dataset):
    def __init__(self, checkpoint_path, seed, ps):
        super().__init__()        
        with open(os.path.join(checkpoint_path, 'HyperParam.yml'), 'r') as file:
            HpParams = yaml.safe_load(file)
        self.f_size = HpParams['f_size'] # frame size: the number of frames you want to look backback
        self.dilation = HpParams['dilation'] # distance between two anchors while downsampling along time axis
        self.ps = ps
        dataset_name = HpParams['dataset_name']
        if dataset_name == 'vir_poppy':
            self.anchor = HpParams['anchor']
            video_list = pd.read_hdf(os.path.join(checkpoint_path, str(seed), 'sup.h5'))
            video_list = video_list[video_list['class'] == 'fall_new']
            video_list = video_list[video_list['line'] >= self.f_size + self.ps - 1]
            self.frames_labels_idx = self.get_lables_frames_VP(video_list)
            print(f'ps: {self.ps}, sup {dataset_name} dataset: {len(self.frames_labels_idx)} video clips from {len(video_list)} videos')
        elif dataset_name == 'real_poppy':
            path = os.path.join(os.getcwd(), 'real_poppy')
            with open(os.path.join(checkpoint_path, str(seed), 'sup.pkl'), 'rb') as f:
                num_reps = pk.load(f)
            results, bufs, frames = {}, {}, {}
            for rep in num_reps:
                with open(os.path.join(path, 'results_%d.pkl' % (-rep-1)), 'rb') as f:
                    (_, results[rep], bufs[rep]) = pk.load(f, encoding='latin1') # motor_names, results[rep], bufs[rep]
                with open(os.path.join(path, 'frames_%d.pkl' % (-rep-1)), 'rb') as f:
                    frames[rep] = pk.load(f, encoding='latin1')
            self.frames_labels_idx = self.get_labels_frames_RP(num_reps, results, bufs, frames)
            print(f'ps: {self.ps}, sup {dataset_name} dataset: {len(self.frames_labels_idx)} video clips from {len(num_reps)} videos')

    def __len__(self):
        return len(self.frames_labels_idx)
    
    def __getitem__(self, index):
        video_clip = torch.tensor(self.frames_labels_idx[index][0]).float().permute(3,0,1,2) / 255 # tensor(C, f_size, H, W)
        label = self.frames_labels_idx[index][1]
        idx = self.frames_labels_idx[index][2]
        return video_clip, label, idx
    
    def get_labels_frames_RP(self, num_reps, results, bufs, frames):
        frames_lables_idx = []
        for count, rep in enumerate(num_reps):
            elapsed_rep = []
            frames_temp = []
            actuals = []
            '''
            Near the end there is a transition to lift, and then a transition to kick.
            In practice, these two transitions happened very quickly, because one of Poppy's feet does not touch the ground,
            and it can easily lose balance if it stays on only one foot for too long.
            They were in fact so quick that I could not reliably distinguish them when I recorded successes by hand.
            So these two transitions are lumped together for the purposes of recording success.
            '''
            if results[rep] >= 3:
                checkpoint = results[rep] + 1
                line = (22 + 2 + 3 * (results[rep] - 2)) * 10 - self.ps
            else:
                checkpoint = results[rep]
                line = (11 * results[rep]) * 10 - self.ps

            for k in range(5):
                _, buffers, elapsed = zip(*bufs[rep][k]) # (flag, buffers, elapsed)
                
                # actual labels
                if k < checkpoint:
                    label = 0 # standing/walking
                else:
                    label = 1 # fall
                for _ in range(len(buffers)*10): # 10 waypoints/move
                    actuals.append(label)
                # accumulate elapsed time over multiple waypoints
                for i in range(1, len(elapsed)):
                    elapsed[i][:] = elapsed[i] + elapsed[i-1][-1]
                elapsed = np.concatenate(elapsed)
                elapsed_rep.append(elapsed)
                # video clip
                frames_temp.append(np.concatenate(np.stack(frames[rep][k])))

            actuals = np.array(actuals)
            for i in range(1, len(elapsed_rep)):
                elapsed_rep[i][:] = elapsed_rep[i] + elapsed_rep[i-1][-1]
            elapsed_rep = np.concatenate(elapsed_rep)
            frames_temp = np.concatenate(frames_temp)

            # linear interpolation: frames and positions
            interval = elapsed_rep[-1]/elapsed_rep.shape[0]
            destination_anchor = int(elapsed_rep.shape[0]/self.dilation)
            idx = [interval * (i+1) * self.dilation for i in range(destination_anchor)]
            upper_bound_idx = np.searchsorted(elapsed_rep, idx)
            lower_bound_idx = upper_bound_idx -1

            intpolat_frames, intpolat_actuals = [], []
            # reason for -1: upper_bound_idx[-1] is len(elapsed_rep), which will results in "index exceeds" error
            for i in range(destination_anchor-1):  
                # [frames[lower] * (distance btwn idx[i] and its upper bound) + frames[upper] * (distance btwn idx[i] and its lower bound)]
                new_frame = frames_temp[lower_bound_idx[i]]*(elapsed_rep[upper_bound_idx[i]] - idx[i]) + frames_temp[upper_bound_idx[i]]*(idx[i]-elapsed_rep[lower_bound_idx[i]])
                new_frame /= elapsed_rep[upper_bound_idx[i]] - elapsed_rep[lower_bound_idx[i]]
                intpolat_frames.append(new_frame)

                # actuals[lower_bound_idx[i]] or actuals[higher_bound_idx[i]], which one is closer to idx[i]
                # IMPORTANT INFO: I found labels '0' (walking) in the sup (shift testing) experiments. The way I calculate interpolated acutals might be the reason.
                if elapsed_rep[upper_bound_idx[i]] - idx[i] >= idx[i]-elapsed_rep[lower_bound_idx[i]]:
                    new_actuals = actuals[lower_bound_idx[i]]
                else:
                    new_actuals = actuals[upper_bound_idx[i]]
                intpolat_actuals.append(new_actuals)
                
            # append the last image (frames_temp[-1]) to make up the dropped image
            # Note: frames_temp[-1] is not always frames[rep][-1][-1][-1] since the results[rep] could be smaller than 4
            intpolat_frames.append(frames_temp[-1])

            # append the last position (actuals[-1]) to make up the dropped image
            intpolat_actuals.append(actuals[-1])

            if results[rep] == 4: # a non-fall episode
                anchor = random.randint(self.f_size, line)
                frames_lables_idx.append((np.stack(intpolat_frames[anchor - self.f_size : anchor]), intpolat_actuals[anchor + self.ps - 1], anchor + self.ps - 1))
            else:
                frames_lables_idx.append((np.stack(intpolat_frames[line - self.f_size : line]), intpolat_actuals[line + self.ps - 1], line + self.ps - 1))

        return frames_lables_idx
    
    def get_lables_frames_VP(self, video_list):
        frames_lables_idx = []

        for idx, (folder, clss, line) in enumerate(zip(list(video_list['folder']), list(video_list['class']), list(video_list['line']))):
            
            path = os.path.join(os.getcwd(), 'virtual_poppy', clss, folder)
            sp_anchor = len(glob.glob(os.path.join(path, '*.png')))-1 # stop anchor is the highest index of images of this full video
            frames = []
            indexes = []
            shifted_indexes = []
            anchor = self.anchor
            while anchor <= sp_anchor:
                img = Image.open(os.path.join(path, '{}.png'.format(anchor)))
                img.load()
                img = np.asarray(img.convert('RGB'), dtype='uint8')
                frames.append(img)
                indexes.append(anchor)
                shifted_indexes.append(anchor - self.dilation)
                anchor += self.dilation

            with open(os.path.join(path, 'hc_z.pkl'), 'rb') as f:
                hc_z_coordinates = [coordinate[2] for coordinate in pk.load(f)]
                hc_z_coordinates_shifted = np.array(hc_z_coordinates)[shifted_indexes]
                hc_z_coordinates = np.array(hc_z_coordinates)[indexes]

            # 1 -> fall, 0 -> standing
            label = list((np.absolute(hc_z_coordinates - hc_z_coordinates_shifted)>0.00169).astype(int))
            
            if clss == 'standing': # a non-fall episode
                anchor = random.randint(self.f_size, line - self.ps + 1)
                frames_lables_idx.append((np.stack(frames[anchor - self.f_size : anchor]), label[anchor + self.ps - 1], anchor + self.ps - 1))
            elif clss == 'fall_new':
                frames_lables_idx.append((np.stack(frames[line - self.ps - self.f_size + 1: line - self.ps + 1]), label[line], line)) # vc labeled as "FALL"
                line_StandingInFall = [i + self.f_size + self.ps - 1 for i in range(len(label[self.f_size + self.ps - 1: line])) if label[self.f_size + self.ps - 1: line][i] == 0]
                if len(line_StandingInFall) > 0:
                    line_SIF = random.choice(line_StandingInFall)
                    frames_lables_idx.append((np.stack(frames[line_SIF - self.ps - self.f_size + 1: line_SIF - self.ps + 1]), label[line_SIF], line_SIF)) # vc labeled as "STANDING"

        return frames_lables_idx
 
    def tell_me_HW(self):
        return self.frames_labels_idx[0][0].shape[:3]
    

if __name__ == '__main__':
    
    # dataset = m2_mydataset_sup(os.path.join(os.getcwd(), 'checkpoints', 'gaga_vir_ps1_256'), 573, 1)
    # loader = DataLoader(dataset, batch_size=10)
    # for idx, (vc, label, idx_f) in enumerate(loader):
    #     print('length of dataset: {}, length of dataloader: {}, current batch: {}/{}'.format(len(dataset), len(loader), idx+1, len(loader)))
    #     print(vc.shape, label, idx_f)

    dataset = m2_mydataset(os.path.join(os.getcwd(), 'checkpoints', 'gaga_vir_ps1_256'), 573)
    print(dataset.tell_me_HW(), len(dataset))
    loader = DataLoader(dataset, batch_size=10)
    for idx, (vc, label, idx_f) in enumerate(loader):
        print('length of dataset: {}, length of dataloader: {}, current batch: {}/{}'.format(len(dataset), len(loader), idx+1, len(loader)))
        print(vc.shape, label, idx_f.shape)

    # from hp import hyperparameters_virtual, hyperparameters_real
    # HpParams = hyperparameters_virtual
    # dataset = m1_mydataset(
    #     HpParams['dilation'], 
    #     HpParams['f_size'], 
    #     HpParams['prediction_span'], 
    #     HpParams['dataset_name'],
    #     'training',
    #     os.path.join(os.getcwd(), 'checkpoints', 'gaga_vir_ps1_256', '573'),
    #     anchor = HpParams['anchor']
    #     )
    # print(dataset.tell_me_HW(), len(dataset))
    # loader = DataLoader(dataset, batch_size=10)
    # for idx, (vc, idx_of_start) in enumerate(loader):
    #     print('length of dataset: {}, length of dataloader: {}, current batch: {}/{}'.format(len(dataset), len(loader), idx+1, len(loader)))
    #     print(vc.shape, idx_of_start.shape)

    # dataset = baseline_dataset('real_poppy')
    # print(dataset.tell_me_HW(), len(dataset))
    # loader = DataLoader(dataset, batch_size=100)
    # for idx, (vc, label) in enumerate(loader):
    #     print('length of dataset: {}, length of dataloader: {}, current batch: {}/{}'.format(len(dataset), len(loader), idx+1, len(loader)))
    #     print(vc.shape, label.shape)
    pass