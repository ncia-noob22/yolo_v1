from tqdm import tqdm


def train(trainloader, model, opt, loss, device, **kwargs):
    loop, mean_losses = tqdm(trainloader), []
    for batch, (data, labels) in enumerate(loop):
        data, labels = data.to(device), labels.to(device)

        opt.zero_grad()

        preds = model(data)
        losses = loss(preds, labels)
        mean_losses.append(losses.item())

        losses.backward()
        opt.step()

        loop.set_postfix(loss=losses.item())
    print(f"Mean loss is {sum(mean_losses) / len(mean_losses)} for {batch + 1}th batch")
