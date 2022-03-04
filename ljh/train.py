from tqdm import tqdm


def train(trainloader, model, opt, sched, loss, epoch, device, **kwargs):
    mean_losses = []
    for data, labels in tqdm(trainloader):
        data, labels = data.to(device), labels.to(device)

        opt.zero_grad()

        preds = model(data)
        losses = loss(preds, labels)
        mean_losses.append(losses.item())

        losses.backward()
        opt.step()

    print(f"Mean loss is {sum(mean_losses) / len(mean_losses)} for {epoch + 1}th epoch")
    sched.step()
