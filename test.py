import wandb

def test_wandb():
    try:
        run = wandb.init(
            project="test-connection",
            entity="alex_larionov",
            name="connection-test",
            config={"test": True}
        )
        
        wandb.log({"test_metric": 1.0})
        print("W&B подключение успешно!")
        print(f"Ссылка: https://wandb.ai/alex_larionov/test-connection")
        
        run.finish()
        
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    test_wandb()