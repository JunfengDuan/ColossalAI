# -*- coding: utf-8 -*-
import os
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding
)
from datasets import load_dataset
import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam

NUM_CLASSES=5
NUM_EPOCHS=3
pre_trained_model="./deepseek/deepseek-r1-base"
train_file_path = "./data/train.txt"
test_file_path = "./data/test.json"

# 分布式训练初始化
colossalai.launch_from_torch(
    config={},
    seed=42
)
coordinator = DistCoordinator()


# 1. 数据预处理
def prepare_dataloaders():
    tokenizer = AutoTokenizer.from_pretrained("deepseek/deepseek-r1-base")

    # 示例数据格式：每行{"text": "...", "label": 0}
    dataset = load_dataset("json", data_files={
        "train": train_file_path,
        "test": test_file_path
    })

    def tokenize_func(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
            padding="max_length"
        )

    tokenized_ds = dataset.map(tokenize_func, batched=True)
    tokenized_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader = DataLoader(
        tokenized_ds["train"],
        shuffle=True,
        batch_size=32,
        collate_fn=data_collator
    )

    test_loader = DataLoader(
        tokenized_ds["test"],
        batch_size=32,
        collate_fn=data_collator
    )

    return train_loader, test_loader


# 2. 模型准备
def build_model(num_classes):
    model = AutoModelForSequenceClassification.from_pretrained(
        pre_trained_model,
        num_labels=num_classes
    )
    return model


# 3. 训练配置
def main():
    # 参数设置
    num_classes = NUM_CLASSES  # 根据实际类别数修改
    num_epochs = NUM_EPOCHS
    lr = 2e-5

    # 准备数据
    train_loader, test_loader = prepare_dataloaders()

    # 初始化模型
    model = build_model(num_classes)

    # 配置ColossalAI优化策略，实现显存优化，支持zero级别的显存优化

    plugin = GeminiPlugin(
        precision="fp16",  # 混合精度训练
        initial_scale=2 ** 16,  # 优化器缩放系数
        max_norm=1.0,  # 梯度裁剪
        placement_policy="auto",  # 自动优化显存
        enable_gradient_accumulation=True, # 开启梯度累积
        enable_fused_normalization=False,  # 融合归一化,需要CUDA >= 11.0
        enable_flash_attention=False,  # 启用Flash Attention
    )

    booster = Booster(plugin=plugin)

    # 封装优化器
    optimizer = HybridAdam(
        model.parameters(),
        lr=lr,
        weight_decay=0.01
    )

    # 使用Booster封装组件
    model, optimizer, _, train_loader, test_loader = booster.boost(
        model=model,
        optimizer=optimizer,
        dataloader=train_loader,
        dataloader_val=test_loader
    )

    # 4. 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            inputs = batch["input_ids"].cuda()
            masks = batch["attention_mask"].cuda()
            labels = batch["label"].cuda()

            # 前向传播
            outputs = model(input_ids=inputs, attention_mask=masks, labels=labels)
            loss = outputs.loss

            # 反向传播与优化
            booster.backward(loss, optimizer)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

            # 每100步打印日志
            if batch_idx % 100 == 0 and coordinator.is_master():
                print(f"Epoch {epoch} | Step {batch_idx} | Loss: {loss.item():.4f}")

        # 验证阶段
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["input_ids"].cuda()
                masks = batch["attention_mask"].cuda()
                labels = batch["label"].cuda()

                outputs = model(input_ids=inputs, attention_mask=masks)
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        if coordinator.is_master():
            accuracy = correct / total
            print(f"Epoch {epoch} | Val Acc: {accuracy:.4f}")
            # 保存检查点
            booster.save_model(model, f"epoch{epoch}_acc{accuracy:.2f}.pt")


if __name__ == "__main__":
    main()
