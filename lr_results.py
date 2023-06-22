import torch
from torch import nn
import numpy as np
from torchvision import transforms
from PIL import Image



class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)


class LPRNet(nn.Module):
    def __init__(self, lpr_max_len, class_num, dropout_rate=0.2):
        super(LPRNet, self).__init__()
        self.lpr_max_len = lpr_max_len
        self.class_num = class_num
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 3), stride=(1, 3)),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),  # 2
            nn.MaxPool3d(kernel_size=(1, 5, 3), stride=(1, 1, 1)),
            small_basic_block(ch_in=64, ch_out=128),  # *** 4 ***
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),  # 6
            nn.MaxPool3d(kernel_size=(1, 5, 3), stride=(2, 1, 2)),
            small_basic_block(ch_in=64, ch_out=256),  # 8
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 10
            small_basic_block(ch_in=256, ch_out=512),  # *** 11 ***
            nn.BatchNorm2d(num_features=512),  # 12
            nn.ReLU(),
            small_basic_block(ch_in=512, ch_out=512),
            nn.MaxPool3d(kernel_size=(1, 5, 3), stride=(4, 1, 2)),  # 14
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=(5, 4), stride=1),  # 16
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(),  # 18
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=512, out_channels=class_num, kernel_size=(3, 3)),  # 20
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU(),  # *** 22 ***
        )
        self.container = nn.Sequential(
            nn.Conv2d(in_channels=self.class_num, out_channels=self.class_num, kernel_size=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.container(self.backbone(x))


class lr:
    def __init__(self, ):
        self.data_transform = transforms.Compose([
            # Reseize our images to 64x64
            transforms.Resize(size=(35, 90)),
            transforms.ToTensor()
        ])
        self.CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                      'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                      'T',
                      'U', 'V', 'W',
                      'X', 'Y', 'Z', '.', '-']

        self.CHARS_DICT = {char: i for i, char in enumerate(self.CHARS)}
        self.model = LPRNet(13, len(self.CHARS_DICT))
        self.model.load_state_dict(torch.load('lr_ch.pt', map_location=torch.device('cpu')))

    def get_result(self, image_path):
        # 1. Load in image and convert the tensor values to float32
        with Image.open(image_path) as f:
            target_image = self.data_transform(f)

        # 5. Turn on model evaluation mode and inference mode
        self.model.eval()
        with torch.inference_mode():
            # Add an extra dimension to the image
            target_image = target_image.unsqueeze(dim=0)

            # Make a prediction on image with an extra dimension and send it to the target device
            target_image_pred = self.model(target_image)

        t_logits = target_image_pred.cpu()
        t_logits = t_logits.squeeze()
        t_logits = t_logits.permute(1, 0)
        t_logits = t_logits.detach().numpy()
        predicted = ""
        for j in range(13):
            predicted += self.CHARS[np.argmax(t_logits[j])]

        predicted = predicted.replace("-", "")
        print(predicted)
        return predicted
