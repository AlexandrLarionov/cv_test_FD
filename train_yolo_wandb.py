import wandb
from ultralytics import YOLO
import torch
import numpy as np
from tqdm import tqdm

def calculate_f1(precision, recall):
    """Вычисление F1-score с защитой от деления на ноль"""
    return (2 * precision * recall) / (precision + recall + 1e-16)

def enhanced_yolo_training():
    """Улучшенное обучение YOLO с расширенным логированием в W&B"""
    
    run = None
    try:
        # Инициализация W&B с расширенной конфигурацией
        run = wandb.init(
            project="dish-detection-yolo",
            entity="alex_larionov",
            name="enhanced-training-v2",
            config={
                "epochs": 3,
                "batch_size": 4,
                "img_size": 640,
                "model": "yolo11n.pt",
                "optimizer": "auto",
                "lr0": 0.01
            }
        )
        print("✓ W&B инициализирован")
        
        # Загрузка модели с проверкой CUDA
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = YOLO('yolo11n.pt').to(device)
        print(f"Модель загружена на {device.upper()}")
        
        # Обучение с ручным логированием метрик
        results = model.train(
            data='food_det_dataset/data.yaml',
            epochs=3,
            batch=4,
            imgsz=640,
            device=device,
            verbose=False
        )
        
        # Ручное логирование метрик после обучения
        if hasattr(results, 'results_dict'):
            final_metrics = {
                "mAP50": results.results_dict.get("metrics/mAP50(B)", 0),
                "precision": results.results_dict.get("metrics/precision(B)", 0),
                "recall": results.results_dict.get("metrics/recall(B)", 0),
                "f1": calculate_f1(
                    results.results_dict.get("metrics/precision(B)", 0),
                    results.results_dict.get("metrics/recall(B)", 0)
                ),
                "train/box_loss": results.results_dict.get("train/box_loss", 0),
                "val/box_loss": results.results_dict.get("val/box_loss", 0)
            }
            wandb.log(final_metrics)
            print(f"Финальные метрики: {final_metrics}")
        
        # Дополнительно: логирование примеров предсказаний
        try:
            sample_image = "food_det_dataset/valid/images/0001.jpg"  # Укажите реальный путь
            pred = model.predict(sample_image, save=False)
            wandb.log({"predictions": [wandb.Image(pred[0].plot(), caption="Пример предсказания")]})
        except Exception as e:
            print(f"Не удалось залогировать примеры: {str(e)}")
        
        run.finish()
        print("Обучение и логирование завершены")
        return results
        
    except Exception as e:
        print(f"Критическая ошибка: {str(e)}")
        if run:
            wandb.log({"error": str(e)})
            run.finish()
        return None

if __name__ == "__main__":
    print("=== Улучшенное обучение YOLO с W&B ===")
    enhanced_yolo_training()