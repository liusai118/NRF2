import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns
import json
import os

# 假设 MyDataSet 和 BCL_Network 已经定义在 myDataSet 和 models 模块中
import myDataSet as ms
import load_data as ld
from models import BCL_Network
import util


# 准备数据加载器
def getFullDataLoader(X, y, batch_size):
    full_DataSet = ms.MyDataSet(input=X.reset_index(drop=True), label=y.reset_index(drop=True))
    full_DataLoader = DataLoader(dataset=full_DataSet, batch_size=batch_size, shuffle=False)
    return full_DataLoader


# 获取项目路径
def get_project_path():
    return "./"


# 创建保存图片的文件夹
def create_figure_dir(fold):
    figure_dir = os.path.join(get_project_path(), 'figure', f'fold_{fold}')
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)
    return figure_dir


# 创建保存 JSON 文件的文件夹
def create_json_dir(fold):
    json_dir = os.path.join(get_project_path(), 'data', f'fold_{fold}')
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)
    return json_dir


# 训练函数，返回最优模型的路径
def train(myDataLoader, validate_DataLoader, path, fold, writer):
    best_roc = 0
    train_step = 0  # 用于记录训练步数
    validate_step = 0  # 用于记录验证步数
    train_losses = []
    validate_losses = []  # 初始化 validate_losses
    train_f1_scores = []
    validate_f1_scores = []
    train_accuracies = []
    validate_accuracies = []
    epoch_steps = []  # 用于存储每个 epoch 开始时的步数

    figure_dir = create_figure_dir(fold)
    json_dir = create_json_dir(fold)

    for epoch in range(Epoch):
        epoch_steps.append(train_step)  # 记录每个 epoch 开始时的步数
        model.train()  # 确保模型在训练模式
        epoch_train_loss = 0
        output_list = []
        correct_list = []
        for step, (x, y) in enumerate(myDataLoader):
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = loss_func(output, y)
            optimizer.zero_grad()
            loss.backward()  # 仅在训练模式下调用
            optimizer.step()

            # 记录训练损失时使用训练步数
            epoch_train_loss += loss.item()
            train_step += 1
            # Save the step-based loss for training
            train_losses.append((epoch, train_step, loss.item()))

            # 收集输出和标签用于计算 F1 分数
            output_list += output.cpu().detach().numpy().tolist()
            correct_list += y.cpu().detach().numpy().tolist()

            # 每次训练步骤后，验证模型并记录验证损失
            model.eval()  # 确保模型在评估模式
            with torch.no_grad():  # 确保在验证过程中不计算梯度
                validate_accuracy, ROC, PR, validate_f1, validate_loss, validate_step = validate(
                    validate_DataLoader, epoch, validate_step, validate_losses, figure_dir
                )
            model.train()  # 训练步骤结束后重新切换到训练模式

        avg_train_loss = epoch_train_loss / len(myDataLoader)
        print(f"Epoch {epoch}: Avg. Training Loss: {avg_train_loss}")

        # 计算训练集的 F1 分数
        y_pred_train = (np.array(output_list) > 0.5).astype(int).flatten()
        y_true_train = np.array(correct_list).flatten()
        train_f1 = f1_score(y_true_train, y_pred_train)
        train_f1_scores.append(train_f1)
        print(f"Epoch {epoch}: Training F1 Score: {train_f1}")

        validate_f1_scores.append(validate_f1)
        validate_accuracies.append(validate_accuracy)

        if ROC > best_roc:
            best_roc = ROC
            best_model_name = save_model(model, path, fold, epoch)

    scheduler.step(validate_loss)
    print(f"Best model saved at: {best_model_name}")

    # 保存 F1 分数、准确率和损失用于绘图
    f1_scores = {
        "train": train_f1_scores,
        "validate": validate_f1_scores
    }
    accuracies = {
        "train": train_accuracies,
        "validate": validate_accuracies
    }
    losses = {
        "train": train_losses,
        "validate": validate_losses
    }

    # Save losses in the json_dir
    save_losses(os.path.join(json_dir, f'train_losses_fold_{fold}.csv'), train_losses)
    save_losses(os.path.join(json_dir, f'validate_losses_fold_{fold}.csv'), validate_losses)

    # 保存数据到 JSON 文件
    save_to_json(os.path.join(json_dir, f'f1_scores_fold_{fold}.json'), f1_scores)
    save_to_json(os.path.join(json_dir, f'accuracies_fold_{fold}.json'), accuracies)
    save_to_json(os.path.join(json_dir, f'losses_fold_{fold}.json'), losses)

    return best_model_name, f1_scores, accuracies, losses, epoch_steps


# 验证函数
def validate(myDataLoader, epoch, validate_step, validate_losses, figure_dir, save_csv=False, csv_path="validation_predictions.csv"):
    output_list = []
    output_result_list = []
    correct_list = []
    test_loss = 0

    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 确保在验证过程中不计算梯度
        for step, (x, y) in enumerate(myDataLoader):
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = loss_func(output, y)
            test_loss += float(loss)
            output_list += output.cpu().detach().numpy().tolist()
            output = (output > 0.5).int()
            output_result_list += output.cpu().detach().numpy().tolist()
            correct_list += y.cpu().detach().numpy().tolist()

    y_pred = np.array(output_result_list).flatten()
    y_true = np.array(correct_list).flatten()
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    test_loss /= len(myDataLoader)
    validate_step += 1
    validate_losses.append((epoch, validate_step, test_loss))
    #print(f'Validation: Avg. loss: {test_loss:.4f}, Accuracy: {accuracy:.3f}, F1: {f1:.3f}')
    ROC, PR, F1 = util.draw_ROC_Curve(output_list, output_result_list, correct_list, get_project_path() + '/' + 'test')

    if save_csv:
        save_predictions(csv_path, y_true, y_pred)

    plot_confusion_matrix(y_true, y_pred, epoch, figure_dir)
    plot_roc_curve(y_true, output_list, epoch, figure_dir)

    return accuracy, ROC, PR, F1, test_loss, validate_step


# 预测函数
def predict(model_path, test_loader):
    # 加载模型
    model = BCL_Network().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 切换到评估模式

    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # 将回归值转换为二进制标签
            binary_outputs = (outputs > 0.5).int()
            all_outputs.append(binary_outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # 确保预测结果和标签是1维数组
    return np.concatenate(all_outputs).flatten(), np.concatenate(all_labels).flatten()


# 评估函数
def evaluate(myDataLoader, path, fold, best_model_name):
    model.load_state_dict(torch.load(best_model_name, map_location=device))
    output_list = []
    output_result_list = []
    correct_list = []

    model.eval()
    with torch.no_grad():
        for step, (x, y) in enumerate(myDataLoader):
            x, y = x.to(device), y.to(device)
            output = model(x)
            output_list += output.cpu().detach().numpy().tolist()
            output = (output > 0.5).int()
            output_result_list += output.cpu().detach().numpy().tolist()
            correct_list += y.cpu().detach().numpy().tolist()

    ROC, PR, F1 = util.draw_ROC_Curve(output_list, output_result_list, correct_list,
                                      path + '/validate_params_' + str(fold))
    return ROC, PR, F1


# 获取数据集并进行拆分
def getDataSet(train_index, test_index):
    x_train = X.iloc[train_index]
    y_train = y.iloc[train_index]
    x_test = X.iloc[test_index]
    y_test = y.iloc[test_index]
    x_train_, x_validate_, y_train_, y_validate_ = train_test_split(
        x_train, y_train, test_size=0.125, stratify=y_train, random_state=1)

    x_train_ = x_train_.reset_index(drop=True)
    x_validate_ = x_validate_.reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    y_train_ = y_train_.reset_index(drop=True)
    y_validate_ = y_validate_.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    train_DataSet = ms.MyDataSet(input=x_train_, label=y_train_)
    validate_DataSet = ms.MyDataSet(input=x_validate_, label=y_validate_)
    test_DataSet = ms.MyDataSet(input=x_test, label=y_test)

    train_DataLoader = DataLoader(dataset=train_DataSet, batch_size=Batch_Size, shuffle=True)
    validate_DataLoader = DataLoader(dataset=validate_DataSet, batch_size=test_Batch_Size, shuffle=True)
    test_DataLoader = DataLoader(dataset=test_DataSet, batch_size=test_Batch_Size, shuffle=True)

    return train_DataLoader, validate_DataLoader, test_DataLoader


# 绘制损失曲线
def plot_loss_curve(fold, json_dir, figure_dir, smoothing_window=50):
    train_losses_df = pd.read_csv(os.path.join(json_dir, f'train_losses_fold_{fold}.csv'))
    validate_losses_df = pd.read_csv(os.path.join(json_dir, f'validate_losses_fold_{fold}.csv'))

    steps_train = train_losses_df['Step']
    train_losses = train_losses_df['Loss']
    steps_validate = validate_losses_df['Step']
    validate_losses = validate_losses_df['Loss']

    smoothed_train_losses = train_losses.rolling(window=smoothing_window).mean()
    smoothed_validate_losses = validate_losses.rolling(window=smoothing_window).mean()

    plt.figure(figsize=(10, 5))
    plt.plot(steps_train, smoothed_train_losses, label='Smoothed Training Loss')
    plt.plot(steps_validate, smoothed_validate_losses, label='Smoothed Validation Loss')

    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve for Fold {fold}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(figure_dir, f"loss_curve_fold_{fold}.svg"))
    plt.close()


# 绘制 F1 分数曲线
def plot_f1_scores(f1_scores, fold, figure_dir):
    epochs = range(1, len(f1_scores["train"]) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, f1_scores["train"], label='Training F1 Score')
    plt.plot(epochs, f1_scores["validate"], label='Validation F1 Score')

    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title(f'F1 Score Curve for Fold {fold}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(figure_dir, f"f1_score_curve_fold_{fold}.svg"))
    plt.close()


# 绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred, epoch, figure_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for Epoch {epoch}')
    plt.savefig(os.path.join(figure_dir, f"confusion_matrix_epoch_{epoch}.svg"))
    plt.close()


# 绘制 ROC 曲线
def plot_roc_curve(y_true, y_scores, epoch, figure_dir):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic for Epoch {epoch}')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(figure_dir, f"roc_curve_epoch_{epoch}.svg"))
    plt.close()


# 保存模型
def save_model(model, path, fold, epoch):
    model_name = os.path.join(ld.modelDir, path, f'validate_params_{fold}_{epoch}.pkl')
    torch.save(model.state_dict(), model_name)
    return model_name


# 保存损失
def save_losses(filename, losses):
    df = pd.DataFrame(losses, columns=['Epoch', 'Step', 'Loss'])
    df.to_csv(filename, index=False)


# 保存预测结果
def save_predictions(filename, y_true, y_pred):
    predictions_df = pd.DataFrame({"True Label": y_true, "Predicted Label": y_pred})
    predictions_df.to_csv(filename, index=False)
    print(f"Validation predictions saved to {filename}")


# 保存数据到 JSON 文件
def save_to_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to {filename}")


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    Batch_Size = 64
    test_Batch_Size = 128
    LR = 0.001
    Epoch = 5
    K_Fold = 5
    print("Batch_Size", Batch_Size)
    print("LR", LR)
    print("Epoch", Epoch)
    print("K_Fold", K_Fold)

    file_list = ld.create_list(ld.dataDir)
    file_list.sort()
    file_list = ['Nrf2']

    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    device_ids = list(range(num_gpus))

    writer = SummaryWriter(log_dir='runs/experiment_1')

    for path in file_list:
        all_data = pd.read_csv(ld.dataDir + path + '/all_data.txt', sep='\t')
        print("数据列名：", all_data.columns)
        X = all_data['sequence']
        y = all_data['label']

        kf = StratifiedKFold(n_splits=K_Fold, shuffle=True, random_state=1)
        fold = 1
        roc_total, pr_total, F1_total = 0, 0, 0

        for train_index, validate_index in kf.split(X, y):
            train_DataLoader, validate_DataLoader, test_DataLoader = getDataSet(train_index, validate_index)
            model = BCL_Network().to(device)
            if num_gpus > 1:
                model = nn.DataParallel(model, device_ids=device_ids)

            optimizer = torch.optim.Adam(model.parameters(), lr=LR)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)
            loss_func = nn.BCELoss()

            figure_dir = create_figure_dir(fold)
            json_dir = create_json_dir(fold)
            best_model_name, f1_scores, accuracies, losses, epoch_steps = train(train_DataLoader, validate_DataLoader,
                                                                                path, fold, writer)

            ROC, PR, F1 = evaluate(test_DataLoader, path, fold, best_model_name)
            plot_loss_curve(fold, json_dir, figure_dir)
            plot_f1_scores(f1_scores, fold, figure_dir)

            roc_total += ROC
            pr_total += PR
            F1_total += F1

            validate_losses = []  # 初始化 validate_losses 列表
            validate(validate_DataLoader, Epoch, validate_step=0, validate_losses=validate_losses, figure_dir=figure_dir, save_csv=True, csv_path=f"validation_predictions_fold_{fold}.csv")
            fold += 1
            print("#################################")

        roc_average = roc_total / K_Fold
        pr_average = pr_total / K_Fold
        f1_average = F1_total / K_Fold
        print(path)
        print("平均 ROC:{}\tPR:{}\tF1:{}".format(roc_average, pr_average, f1_average))
        print("#################################")

    for path in file_list:
        all_data = pd.read_csv(ld.dataDir + path + '/test.txt', sep='\t')
        print("数据列名：", all_data.columns)
        X = all_data.iloc[:, 0]  # 假设第一列是 sequence
        y = all_data.iloc[:, 1]  # 假设第二列是 label
        full_DataLoader = getFullDataLoader(X, y, test_Batch_Size)
        predictions, labels = predict(best_model_name, full_DataLoader)

        # 保存预测结果为 CSV 文件
        predictions_df = pd.DataFrame({"True Label": labels, "Predicted Label": predictions})
        predictions_df.to_csv(f"test_predictions_{path}.csv", index=False)
        print(f"Test predictions saved to test_predictions_{path}.csv")

    writer.close()
