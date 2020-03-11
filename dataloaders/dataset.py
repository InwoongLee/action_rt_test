import os
from sklearn.model_selection import train_test_split
import subprocess

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from mypath import Path


class VideoDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            dataset (str): Name of dataset. Defaults to 'ucf101'.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
            preprocess (bool): Determines whether to preprocess dataset. Default is False.
    """

    def __init__(self, dataset='ucf101', split='train', clip_len=16, preprocess=False):
        self.root_dir, self.output_dir = Path.db_dir(dataset)
        folder = os.path.join(self.output_dir, split)
        self.clip_len = clip_len
        self.split = split

        # The following three parameters are chosen as described in the paper section 4.1
        # self.resize_height = 128
        # self.resize_width = 171
        # self.crop_size = 112
        self.crop_height_1st = 540
        self.crop_width_1st = 720
        self.resize_height = 112
        self.resize_width = 112
        self.crop_size = 112

        if not self.check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You need to download it from official website.')

        # if (not self.check_preprocess()) or preprocess:
        print('Preprocessing of {} dataset, this will take long, but it will be done only once.'.format(dataset))
        self.preprocess(dataset=dataset)
        

        
        if dataset == 'ntu60':
            self.fnames, self.label_array = [], []
            for label in sorted(os.listdir(folder)):
                for fname in os.listdir(os.path.join(folder, 'ntu')):
                    self.fnames.append(os.path.join(folder, label, fname))
                    self.label_array.append(int(fname[17:20])-1)
    
            assert len(self.label_array) == len(self.fnames)
            print('Number of {} videos: {:d}'.format(split, len(self.fnames)))

        elif dataset == 'voucher13' or dataset == 'voucher6' or dataset == 'voucher4':
            # print("good")
            self.fnames, self.label_array = [], []
            # print(folder)
            for label in sorted(os.listdir(folder)):
                # print(label)
                for fname in os.listdir(os.path.join(folder, label)):
                    self.fnames.append(os.path.join(folder, label, fname))
                    self.label_array.append(int(label))            

            assert len(self.label_array) == len(self.fnames)
            print('Number of {} videos: {:d}'.format(split, len(self.fnames)))

        else:
            # Obtain all the filenames of files inside all the class folders
            # Going through each class folder one at a time
            self.fnames, labels = [], []
            for label in sorted(os.listdir(folder)):
                for fname in os.listdir(os.path.join(folder, label)):
                    self.fnames.append(os.path.join(folder, label, fname))
                    labels.append(label)

            assert len(labels) == len(self.fnames)
            print('Number of {} videos: {:d}'.format(split, len(self.fnames)))

            # Prepare a mapping between the label names (strings) and indices (ints)
            self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
            # Convert the list of label names into an array of label indices
            self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        if dataset == "ucf101":
            if not os.path.exists('dataloaders/ucf_labels.txt'):
                with open('dataloaders/ucf_labels.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id+1) + ' ' + label + '\n')

        elif dataset == 'hmdb51':
            if not os.path.exists('dataloaders/hmdb_labels.txt'):
                with open('dataloaders/hmdb_labels.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id+1) + ' ' + label + '\n')
        elif dataset == 'ntu60':
            if not os.path.exists('dataloaders/ntu_labels.txt'):
                with open('dataloaders/ntu_labels.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id+1) + ' ' + label + '\n')
        elif dataset == 'voucher13':
            if not os.path.exists('dataloaders/voucher_labels.txt'):
                with open('dataloaders/voucher_labels.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id+1) + ' ' + label + '\n')
        elif dataset == 'voucher6':
            if not os.path.exists('dataloaders/voucher_labels2.txt'):
                with open('dataloaders/voucher_labels2.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id+1) + ' ' + label + '\n')
        elif dataset == 'voucher4' or dataset == 'voucher4_te':
            if not os.path.exists('dataloaders/voucher_labels3.txt'):
                with open('dataloaders/voucher_labels3.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id+1) + ' ' + label + '\n')



    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # Loading and preprocessing.
        buffer = self.load_frames(self.fnames[index])
        print(buffer.shape)
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        print(buffer.shape)
        # buffer = self.crop_sample(buffer, self.clip_len, self.crop_size)
        # print(buffer.shape[0]-self.clip_len+1)
        # buffer = self.crop_aug(buffer, self.clip_len, self.crop_size)
        # print(buffer.shape)
        labels = np.array(self.label_array[index])

        if self.split == 'test':
            # Perform data augmentation
            buffer = self.randomflip(buffer)
            
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        return torch.from_numpy(buffer), torch.from_numpy(labels)

    def check_integrity(self):
        if not os.path.exists(self.root_dir):
            return False
        else:
            return True

    def check_preprocess(self):
        # TODO: Check image size in output_dir
        if not os.path.exists(self.output_dir):
            return False
        elif not os.path.exists(os.path.join(self.output_dir, 'train')):
            return False

        for ii, video_class in enumerate(os.listdir(os.path.join(self.output_dir, 'train'))):
            for video in os.listdir(os.path.join(self.output_dir, 'train', video_class)):
                video_name = os.path.join(os.path.join(self.output_dir, 'train', video_class, video),
                                    sorted(os.listdir(os.path.join(self.output_dir, 'train', video_class, video)))[0])
                image = cv2.imread(video_name)
                if np.shape(image)[0] != self.resize_height or np.shape(image)[1] != self.resize_width:
                    return False
                else:
                    break

            if ii == 10:
                break

        return True

    def preprocess(self, dataset='ucf101'):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            os.mkdir(os.path.join(self.output_dir, 'train'))
            os.mkdir(os.path.join(self.output_dir, 'test'))

        if dataset == 'ntu60':
            for root, dirs, files in os.walk(self.root_dir):
                file_list = files

            # video_files = file_list[:7200]
            video_files = file_list[:56880]

            # CS setting
            train_set = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]
            test_set = [3, 6, 7, 10, 11, 12, 20, 21, 22, 23, 24, 26, 29, 30, 32, 33, 36, 37, 39, 40]

            train = self.ntu_protocol(video_files, train_set, 0)
            test = self.ntu_protocol(video_files, test_set, 0)
            

            # CV setting
            # train_set = [2, 3]
            # test_set = [1]

            # train = self.ntu_protocol(video_files, train_set, 1)
            # test = self.ntu_protocol(video_files, test_set, 1)

            print("number of train samples: ", len(train))
            print("number of test samples: ", len(test))

            train_dir = os.path.join(self.output_dir, 'train', 'ntu')
            test_dir = os.path.join(self.output_dir, 'test', 'ntu')

            if not os.path.exists(train_dir):
                os.mkdir(train_dir)
            if not os.path.exists(test_dir):
                os.mkdir(test_dir)

            for video in train:
                self.process_video(dataset, video, 'ntu', train_dir)

            for video in test:
                self.process_video(dataset, video, 'ntu', test_dir)
        elif dataset == 'voucher13':
            for file in os.listdir(self.root_dir):
                file_path = os.path.join(self.root_dir, file)
                video_files = [name for name in os.listdir(file_path)]
                # print(file)
                # # M300 
                # if file != "12":                    
                #     train = video_files[:200]
                #     test = video_files[200:300]
                # else:                    
                #     train = video_files[:100]
                #     test = video_files[100:150]
                
                # M500
                if file == "5" or file == "6" or file == "10":
                    train = video_files[:200]
                    test = video_files[200:300]
                elif file == "12":
                    train = video_files[:100]
                    test = video_files[100:150]
                else:
                    train = video_files[:400]
                    test = video_files[400:500]

                train_dir = os.path.join(self.output_dir, 'train', file)
                test_dir = os.path.join(self.output_dir, 'test', file)

                if not os.path.exists(train_dir):
                    os.mkdir(train_dir)               
                if not os.path.exists(test_dir):
                    os.mkdir(test_dir)

                for video in train:
                    self.process_video(dataset, video, file, train_dir)

                for video in test:
                    self.process_video(dataset, video, file, test_dir)

            # print(good)
        elif dataset == 'voucher6' or dataset == 'voucher4':
            for file in os.listdir(self.root_dir):
                file_path = os.path.join(self.root_dir, file)
                video_files = [name for name in os.listdir(file_path)]

                # M700               
                train = video_files[:600]
                test = video_files[600:700]

                train_dir = os.path.join(self.output_dir, 'train', file)
                test_dir = os.path.join(self.output_dir, 'test', file)

                if not os.path.exists(train_dir):
                    os.mkdir(train_dir)
                if not os.path.exists(test_dir):
                    os.mkdir(test_dir)

                for video in train:
                    self.process_video(dataset, video, file, train_dir)

                for video in test:
                    self.process_video(dataset, video, file, test_dir)

        elif dataset == 'voucher4_te':
            for root, dirs, files in os.walk(self.root_dir):
                file_list = files
            print(file_list)
            test = file_list            
            test_dir = os.path.join(self.output_dir, 'test', 'voucher')                
        
            if not os.path.exists(test_dir):
                os.mkdir(test_dir)
            else:
                subprocess.run(["rm", "-rf", test_dir])
                os.mkdir(test_dir)

            for video in test:
                self.process_video(dataset, video, 'voucher', test_dir)


        else:
            # Split train/val/test sets
            for file in os.listdir(self.root_dir):
                file_path = os.path.join(self.root_dir, file)
                video_files = [name for name in os.listdir(file_path)]
                print(video_files)

                train_and_valid, test = train_test_split(video_files, test_size=0.2, random_state=42)
                train, val = train_test_split(train_and_valid, test_size=0.2, random_state=42)

                train_dir = os.path.join(self.output_dir, 'train', file)
                val_dir = os.path.join(self.output_dir, 'val', file)
                test_dir = os.path.join(self.output_dir, 'test', file)

                if not os.path.exists(train_dir):
                    os.mkdir(train_dir)
                if not os.path.exists(val_dir):
                    os.mkdir(val_dir)
                if not os.path.exists(test_dir):
                    os.mkdir(test_dir)

                for video in train:
                    self.process_video(dataset, video, file, train_dir)

                for video in val:
                    self.process_video(dataset, video, file, val_dir)

                for video in test:
                    self.process_video(dataset, video, file, test_dir)

        print('Preprocessing finished.')

    def process_video(self, dataset, video, action_name, save_dir):
        # Initialize a VideoCapture object to read video data into a numpy array
        video_filename = video.split('.')[0]
        if not os.path.exists(os.path.join(save_dir, video_filename)):
            os.mkdir(os.path.join(save_dir, video_filename))

        if dataset == 'ntu60' or dataset == 'voucher4_te':
            capture = cv2.VideoCapture(os.path.join(self.root_dir, video))
        else:
            capture = cv2.VideoCapture(os.path.join(self.root_dir, action_name, video))

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Make sure splited video has at least 16 frames
        EXTRACT_FREQUENCY = 4
        if frame_count // EXTRACT_FREQUENCY <= 16:
            EXTRACT_FREQUENCY -= 1
            if frame_count // EXTRACT_FREQUENCY <= 16:
                EXTRACT_FREQUENCY -= 1
                if frame_count // EXTRACT_FREQUENCY <= 16:
                    EXTRACT_FREQUENCY -= 1

        count = 0
        i = 0
        retaining = True

        while (count < frame_count and retaining):
            retaining, frame = capture.read()
            if frame is None:
                continue

            if count % EXTRACT_FREQUENCY == 0:
                if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                # if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                #     frame = frame[int(frame_height/2-self.crop_height_1st/2):int(frame_height/2+self.crop_width_1st/2), int(frame_width/2-self.crop_height_1st/2):int(frame_width/2+self.crop_width_1st/2)]
                #     frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(i))), img=frame)
                i += 1
            count += 1

        # Release the VideoCapture once it is no longer needed
        capture.release()

    def ntu_protocol(self, file_list, cs_or_cv_set, cs_or_cv):

        filtered_file_list = []

        if cs_or_cv == 1:
            # CAMERA
            for fileNo in range(len(file_list)):
                for camNo in cs_or_cv_set:
                    # print file_list[fileNo][1:4]
                    if int(file_list[fileNo][5:8]) == camNo:
                        filtered_file_list.append(file_list[fileNo])

        else:
            # SUBJECT
            for fileNo in range(len(file_list)):
                for subjNo in cs_or_cv_set:
                    # print file_list[fileNo][1:4]
                    if int(file_list[fileNo][9:12]) == subjNo:
                        filtered_file_list.append(file_list[fileNo])

        return filtered_file_list

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer


    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame
        # print(buffer.shape)

        return buffer

    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        if buffer.shape[0] > clip_len:
            pass
        else:
            temp = np.zeros([clip_len+1, buffer.shape[1], buffer.shape[2], buffer.shape[3]], np.dtype('float32'))
            temp[clip_len+1-buffer.shape[0]:clip_len+1, :, :, :] = buffer
            buffer = temp
        time_index = np.random.randint(buffer.shape[0] - clip_len)
        # time_index = buffer.shape[0] - clip_len
        # time_index = 0
        # Randomly select start indices in order to crop the video
        # height_index = np.random.randint(buffer.shape[1] - crop_size)
        # width_index = np.random.randint(buffer.shape[2] - crop_size)
        height_index = 0
        width_index = 0

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        # print(buffer.shape)
        buffer = buffer[time_index:time_index + clip_len,
                  height_index:height_index + crop_size,
                  width_index:width_index + crop_size, :]
        # print(buffer.shape)
        # buffer = buffer[time_index:time_index + clip_len,
        #           0:crop_size, 0:crop_size, :]

        return buffer

    def crop_sample(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        mul_num = 8
        if buffer.shape[0] > (clip_len):
            pass
        else:
            temp = np.zeros([clip_len+1, buffer.shape[1], buffer.shape[2], buffer.shape[3]], np.dtype('float32'))
            temp[clip_len+1-buffer.shape[0]:clip_len+1, :, :, :] = buffer
            buffer = temp
        time_index = np.random.randint(buffer.shape[0] - clip_len)
        # time_index = buffer.shape[0] - clip_len
        # time_index = 0
        # Randomly select start indices in order to crop the video
        # height_index = np.random.randint(buffer.shape[1] - crop_size)
        # width_index = np.random.randint(buffer.shape[2] - crop_size)
        height_index = 0
        width_index = 0

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        # print(time_index, 2, (time_index + clip_len*2))
        # print(buffer.shape)
        # temp = np.zeros([clip_len, buffer.shape[1], buffer.shape[2], buffer.shape[3]], np.dtype('float32'))
        # count = 0
        # print(time_index, mul_num, time_index+clip_len*mul_num)
        # for i in range(time_index, mul_num, time_index + clip_len*mul_num):
        #     print(i)
        #     temp[count, height_index:height_index + crop_size, width_index:width_index + crop_size, :] = buffer[i,
        #           height_index:height_index + crop_size,
        #           width_index:width_index + crop_size, :]
        #     count += 1
        # buffer = temp
        buffer = buffer[time_index:time_index + clip_len:mul_num,
                  height_index:height_index + crop_size,
                  width_index:width_index + crop_size, :]
        # print(buffer.shape)
        # buffer = buffer[time_index:time_index + clip_len,
        #            0:crop_size, 0:crop_size, :]

        return buffer




if __name__ == "__main__":
    from torch.utils.data import DataLoader
    train_data = VideoDataset(dataset='ucf101', split='test', clip_len=8, preprocess=False)
    train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=4)

    for i, sample in enumerate(train_loader):
        inputs = sample[0]
        labels = sample[1]
        print(inputs.size())
        print(labels)

        if i == 1:
            break
