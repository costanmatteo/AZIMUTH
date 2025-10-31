"""
Example: Training uncertainty predictor with process-based configuration

This example demonstrates how to use the automatic process-based data loading
system to train an uncertainty quantification model.

Usage:
    python examples/train_with_process.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.data import load_process_data, DataPreprocessor, MachineryDataset
from src.models.uncertainty_nn import create_medium_uncertainty_model, GaussianNLLLoss
from torch.utils.data import DataLoader


def main():
    print("="*70)
    print("UNCERTAINTY PREDICTOR - Process-Based Training Example")
    print("="*70)

    # ==========================================================================
    # STEP 1: Configure your process
    # ==========================================================================

    # Select your manufacturing process
    process_type = 'laser'  # Options: 'laser', 'plasma', 'galvanic', 'multibond', 'microetch'

    # Define which columns should be predicted (targets/outputs)
    # Leave empty [] to see available columns
    output_columns = []  # Example: ['Temperature', 'Quality_Score']

    # Data directory
    data_dir = 'src/data/raw'

    # ==========================================================================
    # STEP 2: Load data with automatic column mapping
    # ==========================================================================

    try:
        X, y, column_info = load_process_data(
            process_name=process_type,
            data_dir=data_dir,
            output_columns=output_columns if output_columns else None
        )

        print(f"Successfully loaded data!")
        print(f"Input shape: {X.shape}")
        print(f"Output shape: {y.shape}")

    except ValueError as e:
        print(f"\n{e}\n")
        print("Please update the 'output_columns' list above with your target columns.")
        return
    except FileNotFoundError as e:
        print(f"\n{e}\n")
        print(f"Make sure your CSV file exists in {data_dir}")
        return

    # ==========================================================================
    # STEP 3: Preprocess data
    # ==========================================================================

    preprocessor = DataPreprocessor(scaling_method='standard')
    X_scaled, y_scaled = preprocessor.fit_transform(X, y)

    # Split into train/val/test
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(
        X_scaled, y_scaled,
        train_size=0.7,
        val_size=0.15,
        test_size=0.15
    )

    # ==========================================================================
    # STEP 4: Create datasets and dataloaders
    # ==========================================================================

    train_dataset = MachineryDataset(X_train, y_train)
    val_dataset = MachineryDataset(X_val, y_val)
    test_dataset = MachineryDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # ==========================================================================
    # STEP 5: Create model
    # ==========================================================================

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    model = create_medium_uncertainty_model(
        input_dim=input_dim,
        output_dim=output_dim,
        dropout_rate=0.2
    )

    print(f"\nModel created:")
    print(f"  Input features: {input_dim}")
    print(f"  Output features: {output_dim}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters())}")

    # ==========================================================================
    # STEP 6: Training setup
    # ==========================================================================

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    criterion = GaussianNLLLoss(variance_penalty_alpha=0.2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

    print(f"\nTraining on: {device}")
    print(f"Loss function: Gaussian NLL with alpha=0.2")

    # ==========================================================================
    # STEP 7: Training loop (simplified example)
    # ==========================================================================

    num_epochs = 50  # Reduced for example

    print(f"\nStarting training for {num_epochs} epochs...")
    print("-" * 70)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass
            optimizer.zero_grad()
            mean, variance = model(X_batch)
            loss = criterion(mean, variance, y_batch)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        if (epoch + 1) % 10 == 0:
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(device)
                    y_batch = y_batch.to(device)

                    mean, variance = model(X_batch)
                    loss = criterion(mean, variance, y_batch)
                    val_loss += loss.item()

            val_loss /= len(val_loader)

            print(f"Epoch [{epoch+1}/{num_epochs}] "
                  f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    print("-" * 70)
    print("Training complete!")

    # ==========================================================================
    # STEP 8: Save model and preprocessor
    # ==========================================================================

    os.makedirs('checkpoints', exist_ok=True)

    # Save model
    model_path = f'checkpoints/model_{process_type}.pt'
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")

    # Save preprocessor
    scaler_path = f'checkpoints/scaler_{process_type}.pkl'
    preprocessor.save_scalers(scaler_path)

    # Save column information for inference
    import json
    info_path = f'checkpoints/columns_{process_type}.json'
    with open(info_path, 'w') as f:
        json.dump(column_info, f, indent=2)
    print(f"Column info saved to: {info_path}")

    print("\n" + "="*70)
    print("Process-based training example completed successfully!")
    print("="*70)


if __name__ == '__main__':
    main()
