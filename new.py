import wandb
from ultralytics import YOLO
import torch
import numpy as np
import os
from tqdm import tqdm

def calculate_f1(precision, recall):
    return (2 * precision * recall) / (precision + recall + 1e-16)

def enhanced_yolo_training():
    run = None
    try:
        # Инициализация W&B
        run = wandb.init(
            project="dish-detection-yolo",
            entity="alex_larionov",
            name="enhanced-training-v3",
            config={
                "epochs": 2,  # Исправлено на 3 эпохи
                "batch_size": 4,
                "img_size": 640,
                "model": "yolo11n.pt",
                "optimizer": "auto",
                "lr0": 0.01
            }
        )
        
        print("W&B инициализирован")
        
        # Загрузка модели
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = YOLO('yolo11n.pt').to(device)
        print(f"Модель загружена на {device.upper()}")
        
        # Создаем папку для сохранения метрик
        os.makedirs("metrics", exist_ok=True)
        
        # Обучение с автоматическим логированием через W&B
        results = model.train(
            data='food_det_dataset/data.yaml',
            epochs=3,  # Обучаем сразу 3 эпохи
            batch=4,
            imgsz=640,
            device=device,
            project="dish-detection-yolo",
            name="train-run",
            verbose=True
        )
        
        # Логирование финальных метрик
        if hasattr(results, 'results_dict'):
            metrics = {
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
            
            wandb.log(metrics)
            print("\n Финальные метрики:")
            print(f"mAP50: {metrics['mAP50']:.4f} | F1: {metrics['f1']:.4f}")
        
        # Логирование финальной модели и примеров
        model.save("yolo11n_trained.pt")
        wandb.save("yolo11n_trained.pt")
        
        try:
            sample_image = "food_det_dataset/valid/images/0001.jpg"
            pred = model.predict(sample_image, save=True)
            wandb.log({
                "predictions": wandb.Image(pred[0].plot(), caption="Пример предсказания"),
                "model": wandb.Artifact("trained_model", type="model")
            })
        except Exception as e:
            print(f"Ошибка логирования предсказаний: {str(e)}")
        
        run.finish()
        print("\n✓ Обучение завершено")
        return results
        
    except Exception as e:
        print(f"✗ Ошибка: {str(e)}")
        if run:
            wandb.log({"error": str(e)})
            run.finish()
        return None

if __name__ == "__main__":
    print("=== Улучшенное обучение YOLO с W&B ===")
    enhanced_yolo_training()