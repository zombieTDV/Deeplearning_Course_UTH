def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10, verbose=True):
    train_losses = []
    for epoch in range(num_epochs):
        epoch_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(epoch_loss)
        if verbose:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')
    return train_losses
