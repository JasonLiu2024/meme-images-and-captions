import torch
import numpy as np
from torch.utils.data import dataset, sampler, dataloader
from numpy.typing import NDArray

torch.manual_seed(0)
np.random.seed(0)

"""Schema"""
# all_image_embeddings
# 1) at inference time, models have seen test images
#   train data: first T% of captions for ea image
#   valid data: first V% of captions for ea image
#   ...
# 2) model have unseen images
#   train data: first T% of images, all their captions
#   valid data: first V% of images, all of their captions

class Meme_n_Caption_Data_Manager:
    def __init__(self,
        FILENAME_all_image_embeddings:str, FILENAME_all_caption_embeddings:str,
        FILENAME_MAP_image_index_TO_caption_indices:str):
        self.all_image_embeddings:torch.Tensor = torch.load(
            f=FILENAME_all_image_embeddings)
        self.all_caption_embeddings:torch.Tensor = torch.load(
            f=FILENAME_all_caption_embeddings)
        self.MAP_image_index_TO_caption_indices:dict[int, NDArray] = np.load(
            file=FILENAME_MAP_image_index_TO_caption_indices, allow_pickle=True)

    def get_dataloader(self, batch_size : int,
            RATIO_negative_samples_over_positive_samples : float,
            INDICES_usable_captions: list[int]) -> dataloader.DataLoader:
        return dataloader.DataLoader(
            dataset=Meme_n_Caption_Dataset_Standard(
                RATIO_negative_samples_over_positive_samples,
                INDICES_usable_captions,
                self.MAP_image_index_TO_caption_indices,
                self.all_image_embeddings, self.all_caption_embeddings), 
            drop_last=True, batch_size=batch_size,)

class Meme_n_Caption_Dataset_Standard(dataset.Dataset):
    """
    `__getitem__()`: (i:image, c: caption, l:label) pair; c's matching image may != i\n
    l ∈ {0: c and i match, 1: they don't match}
    \nAssume SAME number of matching captions for each image, 
    \nso the same selection indices are used across all images
    \n`all_caption_embeddings`: 
    \n\tfor example, entry {x, NDArray(y, z, ...)} means 
    \n\tthat the matching captions of xth image (from list of all images)
    \n\tare the yth, zth, ... caption (from list of all captions)
    """
    def __init__(self, RATIO_negative_samples_over_positive_samples: float,
            INDICES_usable_captions: list[int],
            MAP_image_index_TO_caption_indices: dict[int, NDArray],
            all_image_embeddings: torch.Tensor, all_caption_embeddings: torch.Tensor):
        super().__init__()
        # let's say we have M images
        # for each image, we have N captions
        # for train/validat/test-ing, we select K of those captions
        # so in total, we have M * K images
        # we also generate negative samples (see below)
        COUNT_OF_image_embeddings = all_image_embeddings.shape[0]
        self.length = int(len(INDICES_usable_captions) * COUNT_OF_image_embeddings * (1 + RATIO_negative_samples_over_positive_samples))
        image_embeddings : list[torch.Tensor] = []
        caption_embeddings : list[torch.Tensor]= []
        labels : list[torch.Tensor] = []
        # load matching (i, c, l=1) pairs, note omission of '_embedding' in index variable names for convenience
        for image_index in range(COUNT_OF_image_embeddings):
            for caption_index in MAP_image_index_TO_caption_indices[image_index][INDICES_usable_captions]:
                image_embeddings.append(all_image_embeddings[image_index])
                caption_embeddings.append(all_caption_embeddings[caption_index])
                labels.append(torch.tensor(1, dtype=torch.float))
        # generate NON-matching (i, c, l=0) pairs
        COUNT_OF_non_matching_pairs = int(len(INDICES_usable_captions) * COUNT_OF_image_embeddings * RATIO_negative_samples_over_positive_samples)
        INDICES_random_selection_of_images = np.random.choice(a=list(range(COUNT_OF_image_embeddings)), size=COUNT_OF_non_matching_pairs)
        INDICES_random_selection_of_captions = np.random.randint(size=(COUNT_OF_non_matching_pairs), low=0, high=COUNT_OF_image_embeddings - 1)
        for image_index, caption_index in zip(INDICES_random_selection_of_images, INDICES_random_selection_of_captions):
            image_embeddings.append(all_image_embeddings[image_index])
            # I get the next image in the list, using % to wrap the indexing
            image_index_of_a_different_image = (image_index + 1) % (COUNT_OF_image_embeddings - 1)
            caption_embeddings.append(all_caption_embeddings[
                MAP_image_index_TO_caption_indices[image_index_of_a_different_image][INDICES_usable_captions][caption_index]])
            labels.append(torch.tensor(0, dtype=torch.float))

        self.image_embeddings = torch.stack(image_embeddings)
        self.caption_embeddings = torch.stack(caption_embeddings)
        self.labels = torch.stack(labels)
    def __getitem__(self, index : int):
        return self.image_embeddings[index], self.caption_embeddings[index], self.labels[index]
    def __len__(self, ):
        return self.length
