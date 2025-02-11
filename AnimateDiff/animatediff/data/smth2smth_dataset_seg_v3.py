import rootutils
rootutils.setup_root(search_from=__file__, indicator="app.py", pythonpath=True)

import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from animatediff.utils.util import zero_rank_print

import os, io, csv, math, random
import numpy as np
from einops import rearrange
from decord import VideoReader
import json

class Something_v2_seg_v3(Dataset):
    def __init__(
            self,
            csv_path, video_folder,
            sample_size=256, sample_stride=4, sample_n_frames=16,
            video_seg_folder='/home/dungnt206/workspace/data/smth2smth_seg/hand.',
            video_depth_folder='/home/dungnt206/workspace/data/something_something_v2/data/gray_depth-20bn-something-something-v2',
            is_image=False,
        ):
        zero_rank_print(f"loading annotations from {csv_path} ...")
        
        sample_n_frames = int(sample_n_frames/2)
        
        self.dataset = []
        with open(csv_path, 'r') as f:
            annotations = json.load(f)

            for item in annotations:
                # print(item)
                video_id = item['id']
                template = item['template']
                placeholders = item['placeholders']
                
                prompt = template
                for placeholder in placeholders:
                    prompt = prompt.replace('[something]', placeholder, 1)
                    
                self.dataset.append({"videoid": video_id, "name": prompt, "template": template, "placeholders": placeholders})
            
            
            
        self.length = len(self.dataset)
        zero_rank_print(f"data scale: {self.length}")

        self.video_seg_folder = video_seg_folder
        self.video_depth_folder = video_depth_folder
        self.video_folder    = video_folder
        self.sample_stride   = sample_stride
        self.sample_n_frames = sample_n_frames
        self.is_image        = is_image
        
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.pixel_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(sample_size[0]),
            transforms.CenterCrop(sample_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
    
    def get_pixel_values(self, video_folder, videoid, start_idx = -1, binary = False):
        video_dir    = os.path.join(video_folder, f"{videoid}.webm")
        video_reader = VideoReader(video_dir, num_threads=1)
        video_length = len(video_reader)
        
        if not self.is_image:
            clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
            if start_idx == -1:
                start_idx   = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else:
            batch_index = [random.randint(0, video_length - 1)]

        pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous() ## F C H W
        pixel_values = pixel_values / 255.
        del video_reader
        
        #if binary == True:
            #pixel_values = torch.sum(pixel_values, dim=1, keepdim=True)
            # print(pixel_values.shape)
        
        return pixel_values, start_idx
    
    def get_batch(self, idx):
        video_dict = self.dataset[idx]
        videoid, name, _, _ = video_dict['videoid'], video_dict['name'], video_dict['template'], video_dict["placeholders"]
        
        pixel_values_1, start_idx = self.get_pixel_values(self.video_folder, videoid)
        masks, _ = self.get_pixel_values(self.video_seg_folder, videoid + "_highlighted", start_idx, binary=True)
        pixel_values_2, _ = self.get_pixel_values(self.video_depth_folder, videoid, start_idx)


        if pixel_values_2.size() != masks.size():
            pixel_values_2 = torch.nn.functional.interpolate(pixel_values_2, size=masks.size()[2:], mode='nearest')
        
        pixel_values_2 = pixel_values_2 * (masks > 0.5)

        
        pixel_values = torch.concatenate((pixel_values_1, pixel_values_2), dim = 0)
        
        # print(pixel_values_1.shape)
        # print(pixel_values.shape)

        if self.is_image:
            pixel_values = pixel_values[0]
        
        return pixel_values, name

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # while True:
        #     try:
        pixel_values, name = self.get_batch(idx)
                # break

            # except Exception as e:
            #     idx = random.randint(0, self.length-1)

        pixel_values = self.pixel_transforms(pixel_values)
        sample = dict(pixel_values=pixel_values, text=name)
        return sample




if __name__ == "__main__":
    from animatediff.utils.util import save_videos_grid

    dataset = Something_v2_seg_v3(
        csv_path="/home/dungnt206/workspace/data/smth2smth_seg/label/filtered_data.json",
        video_folder="/home/dungnt206/workspace/data/something_something_v2/data/20bn-something-something-v2",
        sample_size=256,
        sample_stride=2, sample_n_frames=32,
        is_image=False,
    )
    # import pdb
    # pdb.set_trace()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=16,)
    for idx, batch in enumerate(dataloader):
        # print(batch["pixel_values"].shape, len(batch["text"]))
        pixel_value = rearrange(batch["pixel_values"], "b f c h w -> b c f h w")[0]
        pixel_value = pixel_value[None, ...]
        text = batch["text"][0]
        output_dir = '/home/dungnt206/workspace/code/AnimateDiff/outputs/20_percent/v5/bins'
        print(pixel_value.shape)
        print(text)
        save_videos_grid(pixel_value, f"{output_dir}/sanity_check/{'-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{idx}'}.gif", rescale=True)
        # for i in range(batch["pixel_values"].shape[0]):
        #     save_videos_grid(batch["pixel_values"][i:i+1].permute(0,2,1,3,4), os.path.join(".", f"{idx}-{i}.mp4"), rescale=True)

