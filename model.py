from torchvision.models import resnet50
import torch as th
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ImageDataset

dst = ImageDataset()
dataloader = DataLoader(dst, batch_size=1024, shuffle=True)

epochs = 200

model = resnet50(weights='ResNet50_Weights.DEFAULT')

# freeze layers except final output
for name, param in model.named_parameters():
    if 'fc' not in name:
        param.requires_grad = False

# change final output to 8 labels
model.fc = nn.Linear(in_features=2048, out_features=8)
model = model.cuda()

loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
optimizer = optim.SGD(
    model.fc.parameters(),
    lr=1e-2,
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=True
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-5)

def train_one_epoch(image_model, loader, criterion, optimizer, scheduler, device):
    image_model.train()

    running_loss = 0.0

    loop = tqdm(loader, leave=True)

    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)

        # ---- Forward ----
        outputs = image_model(images)
        loss = criterion(outputs, labels)

        # ---- Backward ----
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ---- Scheduler (per batch for OneCycle) ----
        scheduler.step()

        # ---- Metrics ----
        running_loss += loss.item()

        loop.set_postfix(
            loss=loss.item(),
        )

    epoch_loss = running_loss / len(loader)

    return epoch_loss



def train_model(model, train_loader, criterion,
                optimizer, scheduler, device, epochs):

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, criterion,
            optimizer, scheduler, device
        )
        print(train_loss)



train_model(model, dataloader ,
            loss_fn, optimizer, scheduler,
            'cuda:0', epochs)


