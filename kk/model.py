import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

CLASSES = {
    'person' : 0,
    'bird' : 1,
    'cat' : 2,
    'cow' : 3,
    'dog' : 4,
    'horse' : 5,
    'sheep' : 6,
    'aeroplane' : 7,
    'bicycle' : 8,
    'boat' : 9,
    'bus' : 10,
    'car' : 11,
    'motorbike' : 12,
    'train' : 13,
    'bottle' : 14,
    'chair' : 15,
    'diningtable' : 16,
    'pottedplant' : 17,
    'sofa' : 18,
    'tvmonitor' : 19,
}

class YOLO(nn.Module):
    def __init__(
        self,
        S=7,
        B=2,
        C=20,
        input_w=448,
        input_h=448,
        lambda_coord=5,
        lambda_noobj=0.5,
        batch_size=4,
        device=torch.device('cuda')
    ):
        super(YOLO, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.device = device
        self.batch_size = batch_size
        self.input_w = input_w
        self.input_h = input_h

        # Convolution Layers
        self.conv1 = nn.Conv2d(3, 192, kernel_size=7, stride=2, padding=3)

        self.conv2 = nn.Conv2d(192, 256, kernel_size=3, padding='same')

        self.conv3 = nn.Conv2d(256, 128, kernel_size=1, padding='same')
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding='same')
        self.conv5 = nn.Conv2d(256, 256, kernel_size=1, padding='same')
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding='same')

        self.conv7 = nn.Conv2d(512, 256, kernel_size=1, padding='same')
        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, padding='same')
        self.conv9 = nn.Conv2d(512, 256, kernel_size=1, padding='same')
        self.conv10 = nn.Conv2d(256, 512, kernel_size=3, padding='same')
        self.conv11 = nn.Conv2d(512, 256, kernel_size=1, padding='same')
        self.conv12 = nn.Conv2d(256, 512, kernel_size=3, padding='same')
        self.conv13 = nn.Conv2d(512, 256, kernel_size=1, padding='same')
        self.conv14 = nn.Conv2d(256, 512, kernel_size=3, padding='same')
        self.conv15 = nn.Conv2d(512, 512, kernel_size=1, padding='same')
        self.conv16 = nn.Conv2d(512, 1024, kernel_size=3, padding='same')

        self.conv17 = nn.Conv2d(1024, 512, kernel_size=1, padding='same')
        self.conv18 = nn.Conv2d(512, 1024, kernel_size=3, padding='same')
        self.conv19 = nn.Conv2d(1024, 512, kernel_size=1, padding='same')
        self.conv20 = nn.Conv2d(512, 1024, kernel_size=3, padding='same')
        self.conv21 = nn.Conv2d(1024, 1024, kernel_size=3, padding='same')
        self.conv22 = nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1)

        self.conv23 = nn.Conv2d(1024, 1024, kernel_size=3, padding='same')
        self.conv24 = nn.Conv2d(1024, 1024, kernel_size=3, padding='same')

        # Fully Connected Layers
        self.fc1 = nn.Linear(7*7*1024, 4096)
        self.fc2 = nn.Linear(4096, self.S * self.S * (5*self.B+self.C))

        # Batch Normalization Layers
        self.batch128 = nn.BatchNorm2d(128)
        self.batch192 = nn.BatchNorm2d(192)
        self.batch256 = nn.BatchNorm2d(256)
        self.batch512 = nn.BatchNorm2d(512)
        self.batch1024 = nn.BatchNorm2d(1024)

        # other layers
        self.maxPool = nn.MaxPool2d(2)
    
    def forward(self, x):
        x = self.batch192(F.leaky_relu(self.conv1(x), negative_slope=0.1))
        x = self.maxPool(x)

        x = self.batch256(F.leaky_relu(self.conv2(x), negative_slope=0.1))
        x = self.maxPool(x)

        x = self.batch128(F.leaky_relu(self.conv3(x), negative_slope=0.1))
        x = self.batch256(F.leaky_relu(self.conv4(x), negative_slope=0.1))
        x = self.batch256(F.leaky_relu(self.conv5(x), negative_slope=0.1))
        x = self.batch512(F.leaky_relu(self.conv6(x), negative_slope=0.1))
        x = self.maxPool(x)

        x = self.batch256(F.leaky_relu(self.conv7(x), negative_slope=0.1))
        x = self.batch512(F.leaky_relu(self.conv8(x), negative_slope=0.1))
        x = self.batch256(F.leaky_relu(self.conv9(x), negative_slope=0.1))
        x = self.batch512(F.leaky_relu(self.conv10(x), negative_slope=0.1))
        x = self.batch256(F.leaky_relu(self.conv11(x), negative_slope=0.1))
        x = self.batch512(F.leaky_relu(self.conv12(x), negative_slope=0.1))
        x = self.batch256(F.leaky_relu(self.conv13(x), negative_slope=0.1))
        x = self.batch512(F.leaky_relu(self.conv14(x), negative_slope=0.1))
        x = self.batch512(F.leaky_relu(self.conv15(x), negative_slope=0.1))
        x = self.batch1024(F.leaky_relu(self.conv16(x), negative_slope=0.1))
        x = self.maxPool(x)

        x = self.batch512(F.leaky_relu(self.conv17(x), negative_slope=0.1))
        x = self.batch1024(F.leaky_relu(self.conv18(x), negative_slope=0.1))
        x = self.batch512(F.leaky_relu(self.conv19(x), negative_slope=0.1))
        x = self.batch1024(F.leaky_relu(self.conv20(x), negative_slope=0.1))
        x = self.batch1024(F.leaky_relu(self.conv21(x), negative_slope=0.1))
        x = self.batch1024(F.leaky_relu(self.conv22(x), negative_slope=0.1))

        x = self.batch1024(F.leaky_relu(self.conv23(x), negative_slope=0.1))
        x = self.batch1024(F.leaky_relu(self.conv24(x), negative_slope=0.1))

        x = nn.Flatten()(x)
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        x = x.view(self.batch_size, self.S, self.S, -1)

        return x
    
    def IOU(self, rect1, rect2):
        """
        Inputs
            rect1, rect2 : torch.tensor with size (BATCH_SIZE, S, S, B)
        Output
            return : torch.tensor with size (BATCH_SIZE, S, S, B)
        """
        rect1_x_max = rect1[..., 0:1] + rect1[..., 2:3] / (self.S * 2)
        rect1_x_min = rect1[..., 0:1] - rect1[..., 2:3] / (self.S * 2)
        rect1_y_max = rect1[..., 1:2] + rect1[..., 3:4] / (self.S * 2)
        rect1_y_min = rect1[..., 1:2] - rect1[..., 3:4] / (self.S * 2)

        rect2_x_max = rect2[..., 0:1] + rect2[..., 2:3] / (self.S * 2)
        rect2_x_min = rect2[..., 0:1] - rect2[..., 2:3] / (self.S * 2)
        rect2_y_max = rect2[..., 1:2] + rect2[..., 3:4] / (self.S * 2)
        rect2_y_min = rect2[..., 1:2] - rect2[..., 3:4] / (self.S * 2)

        overlap_x = torch.minimum(
            torch.maximum(torch.zeros_like(rect1_x_max, dtype=torch.float32).to(self.device), rect1_x_max - rect2_x_min),
            torch.maximum(torch.zeros_like(rect1_x_max, dtype=torch.float32).to(self.device), rect2_x_max - rect1_x_min)
        )
        overlap_y = torch.minimum(
            torch.maximum(torch.zeros_like(rect1_x_max, dtype=torch.float32).to(self.device), rect1_y_max - rect2_y_min),
            torch.maximum(torch.zeros_like(rect1_x_max, dtype=torch.float32).to(self.device), rect2_y_max - rect1_y_min)
        )

        result = overlap_x * overlap_y

        return result

    def yolo_loss(self, target, pred):
        """
        Inputs
            target : torch.tensor with size (BATCH_SIZE, S, S, 5 + self.C)
            pred : torch.tensor with size (BATCH_SIZE, S, S, 5*self.B + self.C)
        Output
            return : YOLO Loss
        """
        # 0. Availability

        # 1. Calculate IOUs and responsibility
        IOUs = torch.empty((self.batch_size, self.S, self.S, self.B), dtype=torch.float32).to(self.device)
        for i in range(self.B):
            IOUs[..., i:i+1] = self.IOU(target[..., :4], pred[..., range(5*i, 5*i + 4)])

        obj_i = torch.any(target, dim=3, keepdim=True).int()

        maximum_iou_values = torch.max(IOUs, dim=3, keepdim=True)[0]
        maximum_iou_mask = torch.ge(IOUs, maximum_iou_values).int()
        obj_ij = maximum_iou_mask * obj_i

        # 2. Calculate Losses
        # 2.1. Localization Loss
        local_x_loss = torch.sum(obj_ij * F.mse_loss(target[..., 0:1], pred[..., [5*i for i in range(self.B)]], reduction='none'))
        local_y_loss = torch.sum(obj_ij * F.mse_loss(target[..., 1:2], pred[..., [5*i+1 for i in range(self.B)]], reduction='none'))
        local_w_loss = torch.sum(obj_ij * F.mse_loss(torch.sqrt(target[..., 2:3]), torch.sqrt(F.sigmoid(pred[..., [5*i+2 for i in range(self.B)]])), reduction='none'))
        local_h_loss = torch.sum(obj_ij * F.mse_loss(torch.sqrt(target[..., 3:4]), torch.sqrt(F.sigmoid(pred[..., [5*i+3 for i in range(self.B)]])), reduction='none'))

        local_loss = self.lambda_coord * (local_x_loss + local_y_loss + local_w_loss + local_h_loss)

        # 2.2. Confidence Loss
        confidence_loss = torch.sum((1 + self.lambda_noobj * (1 - obj_ij)) * F.mse_loss(target[..., 4:5], pred[..., [5*i+4 for i in range(self.B)]], reduction='none'))

        # 2.3. Classification Loss
        class_loss = torch.sum(obj_i * torch.sum((F.mse_loss(target[..., 5:], pred[..., 5*self.B:], reduction='none')), dim=3, keepdim=True))

        # 2.4. Total Loss
        loss = local_loss + confidence_loss + class_loss

        return loss
    
    def _collate_fn(self, batch):
        """
        Inputs
            batch : list of (image, annotation)
        Returns
            (batch_x, batch_y)
            batch_x : torch.Tensor with size [BATCH_SIZE, 3, 448, 448]
            batch_y : torch.Tensor with size [BATCH_SIZE, S, S, 5 + C]
        """
        xs = []
        ys = []
        for image, annotation in batch:
            # 0. Get Image Informations
            image_w, image_h = image.size

            # 1. convert image and append to 'xs'
            x = transforms.PILToTensor()(image) # TODO : torch.transfroms -> Albumentations
            x = transforms.ConvertImageDtype(torch.float32)(x)
            x = transforms.Resize((self.input_w, self.input_h))(x)
            xs.append(x)

            # 2. parse annotation file and append to 'ys'
            # 2.1. for each object in image
            y = torch.zeros(self.S, self.S, 5 + self.C, dtype=torch.float32)
            for obj in annotation['annotation']['object']:
                obj_xmax = int(obj['bndbox']['xmax'])
                obj_ymax = int(obj['bndbox']['ymax'])
                obj_xmin = int(obj['bndbox']['xmin'])
                obj_ymin = int(obj['bndbox']['ymin'])

                # 2.2. normalize x, y, w, h
                obj_x_center = (obj_xmax - obj_xmin) / 2
                obj_y_center = (obj_ymax - obj_ymin) / 2

                obj_x_float = obj_x_center / image_w * S # floating number between 0 and S
                obj_y_float = obj_y_center / image_h * S # floating number between 0 and S

                obj_x = obj_x_float - int(obj_x_float)
                obj_y = obj_y_float - int(obj_y_float)

                obj_w = (obj_xmax - obj_xmin) / image_w
                obj_h = (obj_ymax - obj_ymin) / image_h

                cell_col = int(obj_x_float)
                cell_row = int(obj_y_float)
                
                y[cell_row, cell_col, 0] = obj_x
                y[cell_row, cell_col, 1] = obj_y
                y[cell_row, cell_col, 2] = obj_w
                y[cell_row, cell_col, 3] = obj_h

                # 2.3. confidence
                y[cell_row, cell_col, 4] = 1 # must be multiplied with IoU later

                # 2.4. class vectorize
                class_idx = CLASSES[obj['name']]
                y[cell_row, cell_col, self.B + class_idx] = 1

            # 2.5. append y to 'ys'
            ys.append(y)
        
        # 3. stack 'xs' and 'ys' and return
        return torch.stack(xs, dim=0), torch.stack(ys, dim=0)