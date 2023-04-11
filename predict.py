from cog import BasePredictor, Input, Path
import cv2
import torch

from model import *


MODEL_FILE = 'FT_live.caffemodel.pt'


class Predictor(BasePredictor):
    def setup(self):
        print("Loading model...")
        self.net = Vgg16()
        self.net.load_model(MODEL_FILE)

    def predict(
        self,
        image: Path = Input(
          description="Image to assess"
        ),
    ) -> float:
        """Run a single prediction on the model"""
        # Load image file.
        img = cv2.imread(image)
        img = np.asarray(img)
        x, y = img.shape[0], img.shape[1]

        # Crop patches.
        patch_list = []
        # Randomly crop patches "Num_Patch" times.
        num_Patch = 30
        for j in range(num_Patch):
            x_p = np.random.randint(x - 224)
            y_p = np.random.randint(y - 224)
            patch = img[x_p:(x_p + 224), y_p:(y_p + 224), :]
            patch = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(dim=0).float().cuda()
            patch_list.append(patch)

        # Concat patches at batch_size dim.
        patches = torch.cat(patch_list, dim=0)

        # Get the pred scores.
        score = self.net(patches)  # This network can only accept size(224x224) patch.

        pred = torch.mean(score).item()
        medn = torch.median(score).item()
        return pred
