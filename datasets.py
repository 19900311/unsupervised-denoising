import os
import random

from torch.utils.data import Dataset
from PIL import Image

class ArtifactDataset(Dataset):
    def __init__(self, artifact_dir, no_artifact_dir, transform=None):
        self.artifact_dir = artifact_dir
        self.no_artifact_dir = no_artifact_dir
        self.transform = transform

        self.artifact_images = self._get_all_files(self.artifact_dir)
        self.no_artifact_images = self._get_all_files(self.no_artifact_dir)

        diff = len(self.no_artifact_images) - len(self.artifact_images)
        if diff > 0:
            self.no_artifact_images = random.choices(self.no_artifact_images, k=len(self.artifact_images))
        else:
            self.artifact_images = random.choices(self.artifact_images, k=len(self.no_artifact_images))

        self.artifact_images, self.no_artifact_images = self._replace_matching_patients(self.artifact_images, self.no_artifact_images)

    def _get_all_files(self, directory):
        all_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                all_files.append(os.path.join(root, file))
        return all_files

    def _replace_matching_patients(self, list1, list2):
        for i in range(len(list1)):
            patient1 = os.path.basename(os.path.dirname(list1[i]))
            patient2 = os.path.basename(os.path.dirname(list2[i]))

            while patient1 == patient2:
                list2[i] = random.choice(list2)
                patient2 = os.path.basename(os.path.dirname(list2[i]))

        return list1, list2

    def __len__(self):
        return len(self.artifact_images)

    def __getitem__(self, idx):
        artifact_image = Image.open(self.artifact_images[idx])
        no_artifact_image = Image.open(self.no_artifact_images[idx])

        if self.transform:
            artifact_image = self.transform(artifact_image)
            no_artifact_image = self.transform(no_artifact_image)

        return artifact_image, no_artifact_image