import os
import math
import argparse
from torchvision import transforms, datasets
import torch
import torch.optim as optim
from warmup_scheduler import GradualWarmupScheduler
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import xlwt
from tqdm import tqdm
from Mobilenetv2B import mobilenetv2 as create_model
from utils import read_split_data, train_one_epoch, evaluate


def run_one_train(run_id: int, args):
    best_acc = 0.
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 权重目录
    os.makedirs("./weights", exist_ok=True)

    # 数据增强
    data_transform = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=[0.9, 1.1]),
            transforms.RandomRotation(45),
            transforms.ToTensor(),
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
    }

    # 数据集路径
    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    image_path = os.path.join(data_root, "apple2")
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                              transform=data_transform["val"])

    # DataLoader
    nw = min([os.cpu_count(), args.batch_size if args.batch_size > 1 else 0, 8])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size, shuffle=True,
                                               num_workers=nw)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=args.batch_size, shuffle=False,
                                                  num_workers=nw)

    # ① 每轮重新建模型（保证独立）
    model = create_model(class_num=args.num_classes).to(device)

    # 预训练权重加载（可选）
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # 去掉分类器键（和你原来一致）
        del_keys = ['head.weight', 'head.bias'] if hasattr(model, 'has_logits') else \
                   ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
        for k in del_keys:
            weights_dict.pop(k, None)
        print(model.load_state_dict(weights_dict, strict=False))

    # 优化器 & 学习率调度
    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-9)

    # TensorBoard
    tb_writer = SummaryWriter(comment=f"_run{run_id}")

    # ② 每轮单独 Excel 文件
    book = xlwt.Workbook(encoding='utf-8')
    sheet1 = book.add_sheet(f'Train_data_run{run_id}', cell_overwrite_ok=True)
    header = ['epoch', 'Train_Loss', 'Train_Acc', 'Val_Loss', 'Val_Acc', 'lr', 'Best val Acc']
    for col, name in enumerate(header):
        sheet1.write(0, col, name)

    # --------------- 训练 epoch ---------------
    for epoch in range(args.epochs):
        sheet1.write(epoch + 1, 0, epoch + 1)
        sheet1.write(epoch + 1, 5, str(optimizer.state_dict()['param_groups'][0]['lr']))

        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)
        # val
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=validate_loader,
                                     device=device,
                                     epoch=epoch)
        scheduler.step()

        # 写表
        sheet1.write(epoch + 1, 1, str(train_loss))
        sheet1.write(epoch + 1, 2, str(train_acc))
        sheet1.write(epoch + 1, 3, str(val_loss))
        sheet1.write(epoch + 1, 4, str(val_acc))

        # TensorBoard
        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        # ③ 保存本轮 best
        if val_acc > best_acc:
            best_acc = val_acc
            best_path = f"./weights/best_model_run{run_id}.pth"
            torch.save(model.state_dict(), best_path)
            print(f"[Run {run_id}]  New best acc={best_acc:.4f}  ->  {best_path}")

    # 写 best acc 并保存 Excel
    sheet1.write(1, 6, str(best_acc))
    book.save(f'./Train_data_run{run_id}.xlsx')
    tb_writer.close()
    print(f"==========  Run {run_id} finished. Best Acc = {best_acc:.4f}  ==========")


# --------------------- 主入口 ---------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=7)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--lrf', type=float, default=0.001)
    parser.add_argument('--data-path', type=str, default=r"")
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    args = parser.parse_args()

    # ④ 连续跑 3 轮独立训练
    for r in range(1, 4):
        print(f"\n==========  START Run {r}/3  ==========")
        run_one_train(r, args)
        print(f"==========  END   Run {r}/3  ==========\n")


if __name__ == '__main__':

    main()
