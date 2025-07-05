import wandb
import yaml
from train_yolo_wandb import train_yolo_with_wandb

def run_sweep():
    """
    Функция для запуска sweep оптимизации
    """
    
    # Конфигурация sweep
    sweep_config = {
        'method': 'grid',
        'metric': {
            'goal': 'maximize',
            'name': 'final_mAP50'
        },
        'parameters': {
            'epochs': {'value': 20},
            'batch_size': {'values': [8, 16, 32]},
            'img_size': {'value': 640},
            'learning_rate': {'value': 0.01},
            'lr_final': {'value': 0.01},
            'momentum': {'value': 0.937},
            'weight_decay': {'value': 0.0005},
            'warmup_epochs': {'value': 3},
            'warmup_momentum': {'value': 0.8},
            'warmup_bias_lr': {'value': 0.1},
            'box_loss_gain': {'value': 7.5},
            'cls_loss_gain': {'value': 0.5},
            'dfl_loss_gain': {'value': 1.5},
            'optimizer': {'value': 'SGD'},
            'multi_scale': {'value': True},
            'dropout': {'value': 0.0},
            'cos_lr': {'value': False},
            'close_mosaic': {'value': 10},
            'copy_paste': {'value': 0.0},
            'auto_augment': {'value': 'randaugment'},
            'erasing': {'value': 0.4},
            'crop_fraction': {'value': 1.0},
        }
    }
    
    # Создание sweep
    sweep_id = wandb.sweep(
        sweep_config, 
        project="dish-detection-yolo",
        entity="alex_larionov"
    )
    
    print(f"Sweep создан с ID: {sweep_id}")
    print(f"Ссылка на sweep: https://wandb.ai/alex_larionov/dish-detection-yolo/sweeps/{sweep_id}")
    print(f"Запуск агента...")
    
    # Запуск агента (будет запущено 3 эксперимента с разными batch_size)
    wandb.agent(sweep_id, train_yolo_with_wandb, count=3)

if __name__ == "__main__":
    run_sweep()