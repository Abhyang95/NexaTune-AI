
from pathlib import Path

from src.training.train_qlora import build_trainer


# ============================================================
# Configuration
# ============================================================

FINAL_ADAPTER_DIR = Path("outputs/nexatune-qlora/final_adapter")


# ============================================================
# Main training entry point
# ============================================================

def main():
    print("=" * 70)
    print("NexaTune AI - QLoRA Training Run")
    print("=" * 70)

    print("\nBuilding training pipeline...\n")

    trainer, tokenizer = build_trainer()

    # ========================================================
    # Start training
    # ========================================================

    print("\n" + "=" * 70)
    print("STARTING FULL QLoRA TRAINING")
    print("=" * 70)

    print("\nTraining configuration:")
    print("  Full configured training run")
    print("  No smoke-test run")
    print("  Training executes exactly once")

    train_result = trainer.train()

    # ========================================================
    # Training results
    # ========================================================

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)

    print("\nTraining metrics:")
    print(train_result.metrics)

    # ========================================================
    # Save final LoRA adapter
    # ========================================================

    print("\n" + "-" * 70)
    print("Saving final LoRA adapter")
    print("-" * 70)

    FINAL_ADAPTER_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    trainer.save_model(
        str(FINAL_ADAPTER_DIR)
    )

    tokenizer.save_pretrained(
        str(FINAL_ADAPTER_DIR)
    )

    print(
        f"Final adapter saved to: "
        f"{FINAL_ADAPTER_DIR}"
    )

    # ========================================================
    # Save trainer state
    # ========================================================

    print("\n" + "-" * 70)
    print("Saving trainer state")
    print("-" * 70)

    trainer.save_state()

    print(
        f"Trainer state saved to: "
        f"{trainer.args.output_dir}"
    )

    # ========================================================
    # Final status
    # ========================================================

    print("\n" + "=" * 70)
    print("NexaTune AI QLoRA TRAINING RUN FINISHED")
    print("=" * 70)

    print(f"\nOutput directory : {trainer.args.output_dir}")
    print(f"Final adapter    : {FINAL_ADAPTER_DIR}")

    print("\nDone.")


# ============================================================
# Script entry point
# ============================================================

if __name__ == "__main__":
    main()

