# 保存为 F:\LXP\Project\PythonProject\BatteryLife\train_universal.py

import argparse
import torch
import accelerate
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
import sys
import os
from utils.tools import get_parameter_number
from data_provider.data_factory import data_provider_baseline
import time
import json
import random
import numpy as np
import datetime
import joblib
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error
from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali_baseline, load_content
import importlib
from torch.utils.tensorboard import SummaryWriter  # 添加TensorBoard支持


# 在train_universal.py中，在所有import语句之后，在模型创建之前添加：

# 强制修复MSCA_Plugin的维度问题
def monkey_patch_msca():
    import sys
    sys.path.append('F:\\LXP\\Project\\PythonProject\\BatteryLife')

    # 导入必要的模块
    import torch
    import torch.nn.functional as F
    from models.MSCA_Plugin import MSCABlock

    def fixed_forward(self, x, raw_cycle_data=None):
        """优化效率的forward方法"""
        B, S, D = x.shape

        # Linear transformation path
        linear_features = self.linear_path(x)

        # Multi-scale convolution path（简化版）
        if self.use_conv and raw_cycle_data is not None:
            # 数据格式是 [B, S, C, L]
            B_data, S_data, C, L = raw_cycle_data.shape
            conv_input = raw_cycle_data.view(B * S, C, L)
            conv_features = self.conv_path(conv_input)
            conv_features = F.adaptive_avg_pool1d(conv_features, 1).squeeze(-1)
            conv_features = conv_features.view(B, S, -1)
            fused_features = self.adaptive_gate(linear_features, conv_features)
        else:
            fused_features = linear_features

        # 批量处理attention而不是逐个循环（大幅提速）
        B, S, D = fused_features.shape
        fused_flat = fused_features.view(B * S, D)  # [B*S, D]
        enhanced_flat = self.attention(fused_flat)  # [B*S, D]
        enhanced_features = enhanced_flat.view(B, S, D)  # [B, S, D]

        # Output projection with residual
        out = self.out_proj(enhanced_features)
        residual = self.residual_proj(x)

        return out + residual

    # 替换方法
    MSCABlock.forward = fixed_forward
    print("✓ Applied monkey patch to fix MSCA dimension issue")


# 应用修复
monkey_patch_msca()


# 继续原来的代码...

def list_of_ints(arg):
    return list(map(int, arg.split(',')))


def set_seed(seed):
    accelerate.utils.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_model(model_name):
    """动态导入模型"""
    try:
        # 尝试从models目录导入
        model_module = importlib.import_module(f'models.{model_name}')
        return model_module.Model
    except ImportError:
        print(f"Error: Model {model_name} not found in models directory")
        print("Available models in models directory:")
        models_dir = './models'
        if os.path.exists(models_dir):
            model_files = [f[:-3] for f in os.listdir(models_dir)
                           if f.endswith('.py') and f != '__init__.py']
            for m in sorted(model_files):
                print(f"  - {m}")
        sys.exit(1)


parser = argparse.ArgumentParser(description='Universal Training Script for BatteryLife')

# basic config
parser.add_argument('--task_name', type=str, required=False, default='classification',
                    help='task name')
parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
parser.add_argument('--model', type=str, required=False, default='MSCATransformer', help='model name (e.g., CPMLP, MSCAMLP, MSCATransformer, etc.)')
parser.add_argument('--model_id', type=str, required=False, default='', help='model id (default: same as model name)')
parser.add_argument('--model_comment', type=str, required=False, default='', help='model comment')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# data loader
parser.add_argument('--charge_discharge_length', type=int, default=300, help='resampled length')
parser.add_argument('--dataset', type=str, default='MIX_large', help='dataset')
parser.add_argument('--data', type=str, default='Dataset_original', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset', help='root path')
parser.add_argument('--data_path', type=str, default='', help='data file')
parser.add_argument('--features', type=str, default='MS', help='features')
parser.add_argument('--target', type=str, default='OT', help='target')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='h', help='freq')
parser.add_argument('--checkpoints', type=str, default='./checkpoints', help='checkpoints location')
parser.add_argument('--num_workers', type=int, default=0, help='num workers')

# TensorBoard配置
parser.add_argument('--use_tensorboard', action='store_true', default=True,
                    help='whether to use tensorboard for visualization')
parser.add_argument('--tensorboard_dir', type=str, default='./runs',
                    help='tensorboard log directory')

# forecasting task
parser.add_argument('--early_cycle_threshold', type=int, default=100, help='early cycle threshold')
parser.add_argument('--seq_len', type=int, default=1, help='input sequence length')
parser.add_argument('--pred_len', type=int, default=5, help='prediction length')
parser.add_argument('--label_len', type=int, default=50, help='label length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

# model define
parser.add_argument('--enc_in', type=int, default=3, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=1, help='decoder input size')
parser.add_argument('--c_out', type=int, default=1, help='output size')
parser.add_argument('--d_model', type=int, default=96, help='model dimension')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--lstm_layers', type=int, default=2, help='num of LSTM layers')
parser.add_argument('--e_layers', type=int, default=4, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=3, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=192, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=3, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.15, help='dropout')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
parser.add_argument('--activation', type=str, default='relu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='output attention')
parser.add_argument('--output_num', type=int, default=1, help='output number')
parser.add_argument('--num_class', type=int, default=1, help='number of classes for classification/regression output')

# optimization
parser.add_argument('--des', type=str, default='Exp', help='exp description')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--patience', type=int, default=30, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.00012, help='learning rate')
parser.add_argument('--accumulation_steps', type=int, default=1, help='gradient accumulation')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='cosine', help='learning rate adjust strategy')
parser.add_argument('--weighted_loss', action='store_true', default=False, help='weighted loss')
parser.add_argument('--weighted_sampling', action='store_true', default=False, help='weighted sampling')

# evaluation metrics
parser.add_argument('--alpha1', type=float, default=0.15, help='relative error threshold for 15% accuracy')
parser.add_argument('--alpha2', type=float, default=0.10, help='relative error threshold for 10% accuracy')

args = parser.parse_args()

# Set model_id to model name if not specified
if not args.model_id:
    args.model_id = args.model

if not args.model_comment:
    args.model_comment = args.model

set_seed(args.seed)

# Setup accelerator
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(
    gradient_accumulation_steps=args.accumulation_steps,
    kwargs_handlers=[ddp_kwargs]
)

# Dynamic model loading
print(f'Loading model: {args.model}')
ModelClass = get_model(args.model)
model = ModelClass(args)
print(f'Model {args.model} loaded successfully!')
print(f'Model parameters: {get_parameter_number(model)}')

# Save path
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
save_path = os.path.join(args.checkpoints, f'{args.model_id}_{timestamp}')
if not os.path.exists(save_path) and accelerator.is_main_process:
    os.makedirs(save_path)
print(f'Save path: {save_path}')

# 初始化TensorBoard writer
writer = None
if args.use_tensorboard and accelerator.is_main_process:
    # 创建TensorBoard日志目录
    tb_log_dir = os.path.join(args.tensorboard_dir, f'{args.model_id}_{timestamp}')
    if not os.path.exists(tb_log_dir):
        os.makedirs(tb_log_dir)
    writer = SummaryWriter(log_dir=tb_log_dir)
    print(f'TensorBoard log directory: {tb_log_dir}')
    print(f'To view TensorBoard, run: tensorboard --logdir={args.tensorboard_dir}')

    # 记录超参数
    hparams = {
        'model': args.model,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'd_model': args.d_model,
        'e_layers': args.e_layers,
        'dropout': args.dropout,
        'loss': args.loss,
        'lradj': args.lradj,
        'seed': args.seed,
        'early_cycle_threshold': args.early_cycle_threshold,
        'weighted_loss': args.weighted_loss,
        'weighted_sampling': args.weighted_sampling,
    }

    # 写入超参数
    for key, value in hparams.items():
        writer.add_text('Hyperparameters', f'{key}: {value}', 0)

# Data loading - 训练集会创建scaler
print('Loading training data...')
train_data, train_loader = data_provider_baseline(args, 'train')

# 获取训练集创建的scaler
label_scaler = train_data.label_scaler if hasattr(train_data, 'label_scaler') else None
life_class_scaler = train_data.life_class_scaler if hasattr(train_data, 'life_class_scaler') else None

# 保存scaler到checkpoint目录
if label_scaler is not None and accelerator.is_main_process:
    joblib.dump(label_scaler, os.path.join(save_path, 'label_scaler'))
if life_class_scaler is not None and accelerator.is_main_process:
    joblib.dump(life_class_scaler, os.path.join(save_path, 'life_class_scaler'))

# 加载验证集和测试集，传入scaler
print('Loading validation data...')
vali_data, vali_loader = data_provider_baseline(
    args, 'val',
    label_scaler=label_scaler,
    life_class_scaler=life_class_scaler
)

print('Loading test data...')
test_data, test_loader = data_provider_baseline(
    args, 'test',
    label_scaler=label_scaler,
    life_class_scaler=life_class_scaler
)

print(f'Train samples: {len(train_data)}, Val samples: {len(vali_data)}, Test samples: {len(test_data)}')

# Optimizer
model_optim = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

# Learning rate scheduler
if args.lradj == 'constant':
    scheduler = None
elif args.lradj == 'type1':
    scheduler = lr_scheduler.LambdaLR(model_optim, lr_lambda=lambda epoch: 0.95 ** epoch)
elif args.lradj == 'type2':
    scheduler = lr_scheduler.StepLR(model_optim, step_size=10, gamma=0.5)
elif args.lradj == 'cosine':
    scheduler = lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.train_epochs, eta_min=1e-6)
else:
    scheduler = None

# Loss function
if args.loss == 'MSE':
    criterion = nn.MSELoss()
elif args.loss == 'MAE':
    criterion = nn.L1Loss()
else:
    criterion = nn.MSELoss()

# Prepare for distributed training
if scheduler is not None:
    model, model_optim, train_loader, vali_loader, test_loader, scheduler = accelerator.prepare(
        model, model_optim, train_loader, vali_loader, test_loader, scheduler
    )
else:
    model, model_optim, train_loader, vali_loader, test_loader = accelerator.prepare(
        model, model_optim, train_loader, vali_loader, test_loader
    )

# Early stopping
early_stopping = EarlyStopping(patience=args.patience, verbose=True)

# Training loop
train_steps = len(train_loader)
best_val_loss = float('inf')
best_val_acc = 0
best_metrics = {}
global_step = 0  # 全局步数计数器

print(f'\nStarting training for {args.model}...')
print('=' * 80)

# 获取标准化参数用于反标准化
std, mean_value = np.sqrt(train_data.label_scaler.var_[-1]), train_data.label_scaler.mean_[-1]

for epoch in range(args.train_epochs):
    iter_count = 0
    train_loss = []
    batch_losses = []  # 用于记录每个batch的loss

    model.train()
    epoch_time = time.time()

    for i, (cycle_curve_data, curve_attn_mask, labels, life_class, scaled_life_class, weights,
            seen_unseen_ids) in enumerate(train_loader):
        iter_count += 1
        global_step += 1

        with accelerator.accumulate(model):
            model_optim.zero_grad()

            # 将数据移到设备上
            cycle_curve_data = cycle_curve_data.float()
            curve_attn_mask = curve_attn_mask.float()
            labels = labels.float()

            # Forward pass
            outputs = model(cycle_curve_data, curve_attn_mask)

            # 计算损失
            if args.weighted_loss and weights is not None:
                loss = (criterion(outputs, labels) * weights).mean()
            else:
                loss = criterion(outputs, labels)

            # Backward pass
            accelerator.backward(loss)

            if (i + 1) % args.accumulation_steps == 0:
                # Gradient clipping
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                model_optim.step()
                model_optim.zero_grad()

            train_loss.append(loss.item())
            batch_losses.append(loss.item())

            # 记录每个batch的loss到TensorBoard
            if writer is not None and accelerator.is_main_process:
                writer.add_scalar('Loss/train_batch', loss.item(), global_step)

            if (i + 1) % 100 == 0:
                print(f"Epoch: {epoch + 1}, Step: {i + 1}/{train_steps}, Loss: {loss.item():.6f}")

                # 记录每100步的平均loss
                if writer is not None and accelerator.is_main_process and len(batch_losses) > 0:
                    avg_batch_loss = np.mean(batch_losses)
                    writer.add_scalar('Loss/train_avg_100steps', avg_batch_loss, global_step)
                    batch_losses = []  # 重置

    # Validation
    vali_rmse, vali_mae, vali_mape, vali_alpha_acc1, vali_alpha_acc2 = vali_baseline(
        args, accelerator, model, vali_data, vali_loader, criterion, compute_seen_unseen=False
    )

    # Test
    test_rmse, test_mae, test_mape, test_alpha_acc1, test_alpha_acc2, test_unseen_mape, test_seen_mape, \
        test_unseen_alpha_acc1, test_seen_alpha_acc1, test_unseen_alpha_acc2, test_seen_alpha_acc2 = vali_baseline(
        args, accelerator, model, test_data, test_loader, criterion, compute_seen_unseen=True
    )

    val_loss = vali_mape
    test_loss = test_mape
    train_loss_avg = np.average(train_loss)

    # 获取当前学习率
    current_lr = model_optim.param_groups[0]['lr']

    # 记录到TensorBoard
    if writer is not None and accelerator.is_main_process:
        # 损失指标
        writer.add_scalar('Loss/train_epoch', train_loss_avg, epoch + 1)
        writer.add_scalar('Loss/val', val_loss, epoch + 1)
        writer.add_scalar('Loss/test', test_loss, epoch + 1)

        # 验证集指标
        writer.add_scalar('Metrics/val_rmse', vali_rmse, epoch + 1)
        writer.add_scalar('Metrics/val_mae', vali_mae, epoch + 1)
        writer.add_scalar('Metrics/val_mape', vali_mape, epoch + 1)
        writer.add_scalar('Metrics/val_15%_accuracy', vali_alpha_acc1, epoch + 1)
        writer.add_scalar('Metrics/val_10%_accuracy', vali_alpha_acc2, epoch + 1)

        # 测试集指标
        writer.add_scalar('Metrics/test_rmse', test_rmse, epoch + 1)
        writer.add_scalar('Metrics/test_mae', test_mae, epoch + 1)
        writer.add_scalar('Metrics/test_mape', test_mape, epoch + 1)
        writer.add_scalar('Metrics/test_15%_accuracy', test_alpha_acc1, epoch + 1)
        writer.add_scalar('Metrics/test_10%_accuracy', test_alpha_acc2, epoch + 1)

        # Seen/Unseen分组指标
        writer.add_scalar('Metrics/test_seen_mape', test_seen_mape, epoch + 1)
        writer.add_scalar('Metrics/test_unseen_mape', test_unseen_mape, epoch + 1)
        writer.add_scalar('Metrics/test_seen_15%_accuracy', test_seen_alpha_acc1, epoch + 1)
        writer.add_scalar('Metrics/test_unseen_15%_accuracy', test_unseen_alpha_acc1, epoch + 1)
        writer.add_scalar('Metrics/test_seen_10%_accuracy', test_seen_alpha_acc2, epoch + 1)
        writer.add_scalar('Metrics/test_unseen_10%_accuracy', test_unseen_alpha_acc2, epoch + 1)

        # 学习率
        writer.add_scalar('Learning_Rate/lr', current_lr, epoch + 1)

        # 记录时间
        epoch_duration = time.time() - epoch_time
        writer.add_scalar('Time/epoch_duration', epoch_duration, epoch + 1)

        # 记录多个指标的对比图
        writer.add_scalars('Loss_Comparison', {
            'train': train_loss_avg,
            'val': val_loss,
            'test': test_loss
        }, epoch + 1)

        writer.add_scalars('Accuracy_Comparison', {
            'val_15%': vali_alpha_acc1,
            'val_10%': vali_alpha_acc2,
            'test_15%': test_alpha_acc1,
            'test_10%': test_alpha_acc2
        }, epoch + 1)

        writer.add_scalars('Seen_vs_Unseen_MAPE', {
            'seen': test_seen_mape,
            'unseen': test_unseen_mape
        }, epoch + 1)

    if accelerator.is_main_process:
        print(f"\nEpoch: {epoch + 1}/{args.train_epochs}, Time: {time.time() - epoch_time:.2f}s")
        print(f"Train Loss: {train_loss_avg:.6f}, Val MAPE: {val_loss:.6f}, Test MAPE: {test_loss:.6f}")
        print(f"Learning Rate: {current_lr:.8f}")

        val_15_acc = vali_alpha_acc1 / 100.0
        val_10_acc = vali_alpha_acc2 / 100.0
        test_15_acc = test_alpha_acc1 / 100.0
        test_10_acc = test_alpha_acc2 / 100.0
        test_seen_15_acc = test_seen_alpha_acc1 / 100.0
        test_unseen_15_acc = test_unseen_alpha_acc1 / 100.0
        test_seen_10_acc = test_seen_alpha_acc2 / 100.0
        test_unseen_10_acc = test_unseen_alpha_acc2 / 100.0

        print(f"Val  15% Acc: {val_15_acc:.2%}, 10% Acc: {val_10_acc:.2%}")
        print(f"Test 15% Acc: {test_15_acc:.2%}, 10% Acc: {test_10_acc:.2%}")
        print(f"Test Seen/Unseen - 15% Acc: {test_seen_15_acc:.2%}/{test_unseen_15_acc:.2%}, "
              f"10% Acc: {test_seen_10_acc:.2%}/{test_unseen_10_acc:.2%}")

        # Save best model
        if val_15_acc > best_val_acc:
            best_val_acc = val_15_acc
            best_val_loss = val_loss
            best_metrics = {
                'model': args.model,
                'epoch': epoch + 1,
                '15%-accuracy': test_15_acc,
                '10%-accuracy': test_10_acc,
                'mae': test_mae,
                'mape': test_mape,
                'rmse': test_rmse,
                'seen_15%-accuracy': test_seen_15_acc,
                'unseen_15%-accuracy': test_unseen_15_acc,
                'seen_10%-accuracy': test_seen_10_acc,
                'unseen_10%-accuracy': test_unseen_10_acc,
                'seen_mape': test_seen_mape,
                'unseen_mape': test_unseen_mape,
            }

            print(f"\n*** New best model! Val 15% Acc improved to {val_15_acc:.2%} ***")
            print(f"Saving {args.model} model to {save_path}...")

            # 在TensorBoard中标记最佳模型
            if writer is not None:
                writer.add_scalar('Best/val_15%_accuracy', vali_alpha_acc1, epoch + 1)
                writer.add_scalar('Best/test_15%_accuracy', test_alpha_acc1, epoch + 1)

            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)

            # 保存模型
            model_state = {
                'model_name': args.model,
                'model_state_dict': unwrapped_model.state_dict(),
                'epoch': epoch + 1,
                'val_loss': float(val_loss),
                'val_15_acc': float(val_15_acc),
                'test_15_acc': float(test_15_acc),
            }
            torch.save(model_state, os.path.join(save_path, 'model.pt'))

            # 保存为safetensors格式
            try:
                from safetensors.torch import save_file

                save_file(unwrapped_model.state_dict(), os.path.join(save_path, 'model.safetensors'))
            except ImportError:
                pass

            # Save args and metrics
            with open(os.path.join(save_path, 'args.json'), 'w') as f:
                json.dump(vars(args), f, indent=4)

            with open(os.path.join(save_path, 'best_metrics.json'), 'w') as f:
                metrics_to_save = {
                    'model': args.model,
                    'epoch': epoch + 1,
                    'train_loss': float(train_loss_avg),
                    'val_mape': float(val_loss),
                    'val_mae': float(vali_mae),
                    'val_rmse': float(vali_rmse),
                    'val_15_acc_%': float(vali_alpha_acc1),
                    'val_10_acc_%': float(vali_alpha_acc2),
                    'test_mape': float(test_mape),
                    'test_mae': float(test_mae),
                    'test_rmse': float(test_rmse),
                    'test_15_acc_%': float(test_alpha_acc1),
                    'test_10_acc_%': float(test_alpha_acc2),
                    'test_seen_15_acc_%': float(test_seen_alpha_acc1),
                    'test_unseen_15_acc_%': float(test_unseen_alpha_acc1),
                    'test_seen_10_acc_%': float(test_seen_alpha_acc2),
                    'test_unseen_10_acc_%': float(test_unseen_alpha_acc2),
                    'test_seen_mape': float(test_seen_mape),
                    'test_unseen_mape': float(test_unseen_mape),
                }
                json.dump(metrics_to_save, f, indent=4)

        # Early stopping
        early_stopping(epoch + 1, val_loss, vali_mae, test_mae, model, save_path)
        if early_stopping.early_stop:
            print("\nEarly stopping triggered!")
            if writer is not None:
                writer.add_text('Training', f'Early stopping at epoch {epoch + 1}', epoch + 1)
            break

        print("-" * 80)

    # Learning rate adjustment
    if scheduler is not None:
        adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)

# Final results
if accelerator.is_main_process:
    print("\n" + "=" * 80)
    print(f"TRAINING COMPLETED FOR {args.model}!")
    print("=" * 80)
    print(f"Best validation 15% accuracy: {best_val_acc:.2%}")
    print(f"Best validation MAPE: {best_val_loss:.6f}")
    print(f"\nBest test metrics (at best val acc):")
    print(f"  15%-accuracy: {best_metrics.get('15%-accuracy', 0):.2%}")
    print(f"  10%-accuracy: {best_metrics.get('10%-accuracy', 0):.2%}")
    print(f"  MAE: {best_metrics.get('mae', 0):.4f}")
    print(f"  MAPE: {best_metrics.get('mape', 0):.4f}")
    print(f"  RMSE: {best_metrics.get('rmse', 0):.4f}")
    print(f"  Seen 15% Acc: {best_metrics.get('seen_15%-accuracy', 0):.2%}")
    print(f"  Unseen 15% Acc: {best_metrics.get('unseen_15%-accuracy', 0):.2%}")
    print(f"\nModel saved at: {save_path}")

    # 在TensorBoard中记录最终结果
    if writer is not None:
        # 记录最终的最佳指标
        writer.add_hparams(
            hparam_dict={
                'model': args.model,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'd_model': args.d_model,
                'e_layers': args.e_layers,
                'dropout': args.dropout,
            },
            metric_dict={
                'hparam/best_val_15%_acc': best_val_acc * 100,
                'hparam/best_val_mape': best_val_loss,
                'hparam/best_test_15%_acc': best_metrics.get('15%-accuracy', 0) * 100,
                'hparam/best_test_10%_acc': best_metrics.get('10%-accuracy', 0) * 100,
                'hparam/best_test_mape': best_metrics.get('mape', 0),
                'hparam/best_epoch': best_metrics.get('epoch', 0),
            }
        )

        # 写入最终总结
        summary_text = f"""
        Training Summary for {args.model}:
        - Best Epoch: {best_metrics.get('epoch', 0)}
        - Best Val 15% Acc: {best_val_acc:.2%}
        - Best Test 15% Acc: {best_metrics.get('15%-accuracy', 0):.2%}
        - Best Test MAPE: {best_metrics.get('mape', 0):.4f}
        - Model saved at: {save_path}
        """
        writer.add_text('Summary', summary_text, 0)

        # 关闭writer
        writer.close()
        print(f"TensorBoard logs saved. Run 'tensorboard --logdir={args.tensorboard_dir}' to view.")

    print("=" * 80)