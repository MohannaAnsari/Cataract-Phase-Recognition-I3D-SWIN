import torch, time
import torch.nn as nn
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from i3d_model import build_i3d
from dataset_clip import train_loader, val_loader, class_counts


def run_training(num_epochs=40, lr= 5e-5, patience=10):
    # quick sanity check of label ranges
    # for name, loader in [("train", train_loader), ("val", val_loader)]:
    #     ys = []
    #     for _, y in loader:
    #         ys.append(y)
            
    #     y = torch.cat(ys) if isinstance(ys, list) else ys
    #     print(f"{name} min/max labels = {int(y.min())}, {int(y.max())}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_i3d(num_classes=10, pretrained=True).to(device)
    model.load_state_dict(torch.load("outputs/models/i3d_best_4.pth", map_location=device))

    x, y = next(iter(val_loader))
    x = x.to(device)
    with torch.no_grad():
        preds = model(x).softmax(1)
    print("Sample preds:", preds.argmax(1)[:10].cpu().tolist())
    print("The model loaded successfully!")

    # --- optional partial freezing (helps reduce overfitting on small datasets) ---
    for p in model.parameters():
        p.requires_grad = True


    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # label smoothing combats overconfidence
    
    # --- Weighted cross-entropy to handle phase imbalance ---
    class_cnts = torch.tensor(class_counts, dtype=torch.float32)
    print("Class counts:", class_cnts)
    weights = 1.0 / class_cnts
    weights = weights / weights.sum() * len(class_cnts)  # normalize for stability

    criterion = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.1)

    # assign smaller LR to earlier blocks
    params = [
        {'params': model.blocks[-1].parameters(), 'lr': lr},
        {'params': model.blocks[-2].parameters(), 'lr': lr * 0.5},
    ]
    optimizer = AdamW(params, weight_decay=1e-4)
    optimizer.load_state_dict(torch.load("outputs/models/opt_state_4.pth"))


    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = GradScaler()

    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_path = "outputs/models/i3d_best.pth"
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        print(f"\n🧠 Epoch {epoch}/{num_epochs}")
        model.train()
        correct, total, train_loss_sum = 0, 0, 0.0

        for batch_idx, (X, y) in enumerate(tqdm(train_loader)):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast():
                yhat = model(X)
                loss = criterion(yhat, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            correct += (yhat.argmax(1) == y).sum().item()
            total += y.size(0)
            train_loss_sum += loss.item() * y.size(0)

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  Batch {batch_idx+1}/{len(train_loader)}  "
                      f"Loss={loss.item():.3f}  "
                      f"TrainAcc={correct/total:.3f}")

        train_acc = correct / total
        train_loss = train_loss_sum / total

        # --- Validation ---
        model.eval()
        correct, total, val_loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                yhat = model(X)
                val_loss_sum += criterion(yhat, y).item() * y.size(0)
                correct += (yhat.argmax(1) == y).sum().item()
                total += y.size(0)

        val_acc = correct / total
        val_loss = val_loss_sum / total
        scheduler.step(val_loss)

        print(f"[Epoch {epoch}] "
              f"TrainAcc={train_acc:.3f}, ValAcc={val_acc:.3f}, "
              f"TrainLoss={train_loss:.3f}, ValLoss={val_loss:.3f}")

        # --- Save best model ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            torch.save(optimizer.state_dict(), "outputs/models/opt_state.pth")
            patience_counter = 0
            print("  ✅ New best model saved!")
        else:
            patience_counter += 1

        
        # --- Gradual unfreezing after 5 epochs ---
        if epoch == 10:
            print("🔓 Unfreezing more backbone layers...")
            for p in model.blocks[-3].parameters():
                p.requires_grad = True
            # lower learning rate for all param groups
            for g in optimizer.param_groups:
                g['lr'] *= 0.3

        torch.cuda.empty_cache()

        # --- Early stopping ---
        if patience_counter >= patience:
            print(f"⏹️ Early stopping triggered after {patience} epochs without improvement.")
            break

    print(f"\n🏁 Training completed. Best ValAcc={best_val_acc:.3f}, ValLoss={best_val_loss:.3f}")
    return best_path, best_val_acc

