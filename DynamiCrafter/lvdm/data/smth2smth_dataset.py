import torch
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
# from animatediff.utils.util import zero_rank_print

import os, io, csv, math, random
import numpy as np
from einops import rearrange
from decord import VideoReader
import json



class Something_v2(Dataset):
    def __init__(
            self,
            data_dir,
            meta_path, 
            resolution=256, 
            frame_stride=4, 
            sample_n_frames=16,
            is_image=False,
        ):
        print(f"loading annotations from {meta_path} ...")
        
        self.dataset = []
        with open(meta_path, 'r') as f:
            annotations = json.load(f)

            for item in annotations:
                # print(item)
                video_id = item['id']
                template = item['template']
                placeholders = item['placeholders']
                
                prompt = template
                for placeholder in placeholders:
                    prompt = prompt.replace('[something]', placeholder, 1)
                    
                self.dataset.append({"videoid": video_id, 
                                     "name": prompt, 
                                     "template": template, 
                                     "placeholders": placeholders
                                     })
            
            
            
        self.length = len(self.dataset)
        print(f"data scale: {self.length}")

        self.data_dir    = data_dir
        self.frame_stride   = frame_stride
        self.sample_n_frames = sample_n_frames
        self.is_image        = is_image
        
        resolution = tuple(resolution) if not isinstance(resolution, int) else (resolution, resolution)
        print(resolution)
        self.pixel_transforms = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.Resize(resolution[0]),
            transforms.CenterCrop(resolution),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
    
    def get_batch(self, idx):
        video_dict = self.dataset[idx]
        videoid, name, _, _ = video_dict['videoid'], video_dict['name'], video_dict['template'], video_dict["placeholders"]
        
        video_dir    = os.path.join(self.data_dir, f"{videoid}.webm")
        video_reader = VideoReader(video_dir, num_threads=1)
        video_length = len(video_reader)
        
        if not self.is_image:
            clip_length = min(video_length, (self.sample_n_frames - 1) * self.frame_stride + 1)
            start_idx   = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)
        else:
            batch_index = [random.randint(0, video_length - 1)]

        pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        del video_reader

        if self.is_image:
            pixel_values = pixel_values[0]
        
        return {
            "video": pixel_values, 
            "caption": name,
            "fps": 8,
            "frame_stride": self.frame_stride
        }

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # while True:
        #     try:
        samples = self.get_batch(idx)
        samples['video'] = self.pixel_transforms(samples['video']).permute(1, 0, 2, 3) # fchw -> cfhw
                # break

            # except Exception as e:
            #     idx = random.randint(0, self.length-1)
        
        # pixel_values = self.pixel_transforms(pixel_values)
        # sample = dict(pixel_values=pixel_values, text=name)
        return samples




if __name__ == "__main__":
    # from animatediff.utils.util import save_videos_grid
    print("Running")
    dataset = Something_v2(
        meta_path="/home/ubuntu/video-generation/something_something_v2/data/labels/validation.json",
        data_dir="/home/ubuntu/video-generation/something_something_v2/data/20bn-something-something-v2",
        resolution=[320, 512],
        frame_stride=4, sample_n_frames=16,
        is_image=True,
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=16,)
    for idx, batch in enumerate(dataloader):
        print(batch["video"].shape, len(batch["caption"]))
        # for i in range(batch["pixel_values"].shape[0]):
        #     save_videos_grid(batch["pixel_values"][i:i+1].permute(0,2,1,3,4), os.path.join(".", f"{idx}-{i}.mp4"), rescale=True)

