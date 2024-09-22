from trainers.trainer import Trainer

def main():
    trainer = Trainer(config=None)  # Configuration is handled internally
    trainer.train()

if __name__ == "__main__":
    main()