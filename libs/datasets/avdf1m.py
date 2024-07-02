import os
import json
import numpy as np
import argparse

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from torchvision.io import read_video

from datasets import register_dataset
from data_utils import truncate_feats, padding_audio, padding_video, resize_video

@register_dataset("avdf1m")
class AVDF1M(Dataset):
    def __init__(
        self,
        is_training,     # if in training mode
        split,           # split, a tuple/list allowing concat of subsets
        feat_folder,     # folder for features
        json_file,       # json file for annotations
        feat_stride,     # temporal stride of the feats
        num_frames,      # number of frames for each feat
        max_seq_len,     # maximum sequence length during training
        trunc_thresh,    # threshold for truncate an action segment
        crop_ratio,      # a tuple (e.g., (0.9, 1.0)) for random cropping
        num_classes,     # number of action categories
        file_prefix,     # feature file prefix if any
        file_ext,        # feature file extension if any
        force_upsampling, # force to upsample to max_seq_len
        default_fps = 25,     # default fps
        downsample_rate = 1, # downsample rate for feats
        img_size = 96 # size of each frame
    ):
        # file path
        assert os.path.exists(feat_folder) and os.path.exists(json_file)
        assert crop_ratio == None or len(crop_ratio) == 2
        self.feat_folder = feat_folder
        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ''
        self.file_ext = file_ext
        self.json_file = json_file

        # split / training mode
        self.split = split
        self.is_training = is_training

        # features meta info
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.label_dict = None
        self.crop_ratio = crop_ratio
        self.force_upsampling = force_upsampling
        self.img_size = img_size

        # load database and select the subset
        dict_db = self._load_json_db(self.json_file)
        # "empty" noun categories on epic-kitchens
        assert num_classes == 1 , "Number of fake classifications could be 1 only"
        self.data_list = dict_db

        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'epic-kitchens-100',
            'tiou_thresholds': np.linspace(0.1, 0.5, 5),
            'empty_label_ids': []
        }

        print("{} subset has {} videos".format(self.split,len(self.data_list)))

    def get_attributes(self):
        return self.db_attributes

    def _load_json_db(self, json_file):
        # load database and select the subset
        with open(json_file, 'r') as fid:
            json_db = json.load(fid)

        # fill in the db (immutable afterwards)
        dict_db = tuple()
        for meta in json_db:

            # get fps if available
            if self.default_fps is not None:
                fps = self.default_fps
            else:
                assert False, "Unknown video FPS."

            # get video duration if available
            if 'duration' in meta:
                duration = meta['duration']
            else:
                duration = meta['video_frames'] / fps

            # get annotations if available
            video_labels = int(len(meta['visual_fake_segments'])>0)
            audio_labels = int(len(meta['audio_fake_segments'])>0)

            av_labels = np.array([video_labels,audio_labels])

            # get annotations if available
            if ('fake_segments' in meta) and (len(meta['fake_segments']) > 0):
                valid_acts = meta['fake_segments']
                num_acts = len(valid_acts)
                segments = np.zeros([num_acts, 2], dtype=np.float32)
                labels = np.zeros([num_acts, ], dtype=np.int64)
                for idx, act in enumerate(valid_acts):
                    segments[idx][0] = act[0]
                    segments[idx][1] = act[1]
                    labels[idx] = 0
            else:
                segments = None
                labels = None

            dict_db += ({'id': str(meta['file']).strip(),
                         'fps' : fps,
                         'duration' : duration,
                         'segments' : segments,
                         'audio_frames' : meta['audio_frames'],
                         'video_frames' : meta['video_frames'],
                         'visual_fake_segments' : meta['visual_fake_segments'],
                         'audio_fake_segments' : meta['audio_fake_segments'],
                         'labels' : labels,
                         'av_labels' : av_labels
            }, )

        return dict_db

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        video_item = self.data_list[idx]

        # load audio video

        filename = os.path.join(self.feat_folder, self.split, video_item['id'])
        video, audio, _ = read_video(filename, pts_unit="sec")
        video = video.permute(0, 3, 1, 2) / 255
        audio = audio.permute(1, 0)
        
        video_frames = video.shape[0]
        

        # we support both fixed length features / variable length features
        # case 1: variable length features for training
        if self.feat_stride > 0 and (not self.force_upsampling):
            # var length features
            feat_stride, num_frames = self.feat_stride, self.num_frames
            # only apply down sampling here
            if self.downsample_rate > 1:
                pass #add something
        # case 2: variable length features for input, yet resized for training
        elif self.feat_stride > 0 and self.force_upsampling:
            # feat_stride = float(
            #     (video_frames) * self.feat_stride + self.num_frames
            # ) / self.max_seq_len
            # # center the features
            # num_frames = feat_stride
            feat_stride, num_frames = self.feat_stride, self.num_frames
        # case 3: fixed length features for input
        else:
            # deal with fixed length feature, recompute feat_stride, num_frames
            seq_len = video_frames
            assert seq_len <= self.max_seq_len
            if self.force_upsampling:
                # reset to max_seq_len
                seq_len = self.max_seq_len
            feat_stride = video_item['duration'] * video_item['fps'] / seq_len
            # center the features
            num_frames = feat_stride
        
        feat_offset = 0.5 * num_frames / feat_stride

        # resize the features if needed
        if (video_frames != self.max_seq_len) and self.force_upsampling:
            audio_padding = int(self.max_seq_len/video_item['fps'] * 16000)
            video = padding_video(video, target=self.max_seq_len)
            audio = padding_audio(audio, target=audio_padding)

        elif video_frames>self.max_seq_len:
            video_frames = self.max_seq_len
            audio_padding = int(self.max_seq_len/video_item['fps'] * 16000)
            video = padding_video(video, target=self.max_seq_len)
            audio = padding_audio(audio, target=audio_padding)

        video = resize_video(video, (self.img_size, self.img_size)) #shape = (t c h w)


        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        if video_item['segments'] is not None:
            segments = torch.from_numpy(
                video_item['segments'] * video_item['fps'] / feat_stride - feat_offset
            )
            labels = torch.from_numpy(video_item['labels'])
            # for activity net, we have a few videos with a bunch of missing frames
            # here is a quick fix for training
            if self.is_training:
                vid_len = video_frames + feat_offset
                valid_seg_list, valid_label_list = [], []
                for seg, label in zip(segments, labels):
                    if seg[0] >= vid_len:
                        # skip an action outside of the feature map
                        continue
                    # skip an action that is mostly outside of the feature map
                    ratio = (
                        (min(seg[1].item(), vid_len) - seg[0].item())
                        / (seg[1].item() - seg[0].item())
                    )
                    if ratio >= self.trunc_thresh:
                        valid_seg_list.append(seg.clamp(max=vid_len))
                        # some weird bug here if not converting to size 1 tensor
                        valid_label_list.append(label.view(1))
                segments = torch.stack(valid_seg_list, dim=0)
                labels = torch.cat(valid_label_list)
        else:
            segments, labels = None, None
            
        av_labels = torch.from_numpy(video_item['av_labels'])
        
        # return a data dict
        data_dict = {'video_id'        : video_item['id'],
                     'video'           : video,      # T x C x H x W
                     'audio'           : audio,      # T` x C`
                     'segments'        : segments,   # N x 2
                     'labels'          : labels,     # N
                     'av_labels'       : av_labels,
                     'fps'             : video_item['fps'],
                     'duration'        : video_item['duration'],
                     'actual_frames'   : video_frames,
                     'feat_stride'     : feat_stride,
                     'feat_num_frames' : num_frames}

        # no truncation is needed
        # truncate the features during training
        # if self.is_training and (segments is not None):
        #     data_dict = truncate_feats(
        #         data_dict, self.max_seq_len, self.trunc_thresh, feat_offset, self.crop_ratio
        #     )

        return data_dict
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="VoxCeleb Dataset Test")
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--subset", type=str, default="val")

    args = parser.parse_args()

    json_file_name = args.subset + "_metadata.json"
    json_file = os.path.join(args.data_root, json_file_name)

    dataset = AVDF1M(
        is_training=True, 
        split=args.subset, 
        feat_folder=args.data_root, 
        json_file=json_file, 
        feat_stride=1, 
        num_frames=1, 
        max_seq_len=512, 
        trunc_thresh=0.5, 
        crop_ratio=[0.9, 1.0],
        num_classes=1,
        file_prefix='',
        file_ext='',
        force_upsampling=True,
        default_fps=25,
        downsample_rate=1,
        img_size=96
    )

    first_item = dataset.__getitem__(0)
    print(first_item['video_id'])
    print(first_item['video'].shape)
    print(first_item['audio'].shape)
    print(first_item['segments'])
    print(first_item['actual_frames'])
    print(first_item['feat_stride'])
    print(first_item['feat_num_frames'])

