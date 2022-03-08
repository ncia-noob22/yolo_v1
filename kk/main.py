import torch
import torchvision

from model import YOLO

DATA_DIR = '/data/torch/VOCdetection'

# Learning Configurations
BATCH_SIZE = 4
MAX_EPOCHS = 135
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # Load Dataset
    dataset = torchvision.datasets.VOCDetection(
        root=DATA_DIR,
        year='2012',
        image_set='val',
        download=True
    )

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=YOLO._collate_fn,
    )

    # Initializa Model
    model = YOLO().to(DEVICE)

    # optimizer and lr
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.99, weight_decay=5e-5)

    # start train
    for epoch in range(MAX_EPOCHS):  # loop over the dataset multiple times
        print(f'epoch {epoch + 1}')
        running_loss = 0.0
        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = model.yolo_loss(labels, outputs)
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                print(f'EPOCH : {epoch + 1}, BATCH : {i + 1:5d}, LOSS: {running_loss / 100:.3f}')
                running_loss = 0.0

    print('Finished Training')