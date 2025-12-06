🚀 准备数据集
数据集需按如下结构组织：
your_dataset/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
└── val/
    ├── class1/
    ├── class2/
    └── ...
修改训练脚本中的数据路径（第 45 行）：
image_path = os.path.join(data_root, "your_dataset")  # 修改为你的数据集名称
🎯模型训练
基础训练命令
python train3.py \
    --num_classes 7 \
    --epochs 60 \
    --batch-size 32 \
    --lr 0.01 \
    --device cuda:0
📊训练输出
模型权重：weights/best_model_run{1,2,3}.pth（自动保存每轮最优）
训练日志：Train_data_run{1,2,3}.xlsx
