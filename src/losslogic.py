import torch
import torch.nn as nn

def diagnose_loss_36():
    print("--- DIAGNOSING WHY LOSS IS EXACTLY 36 ---\n")

    # SCENARIO 1: The "Square of Six" (Most Likely for Regression)
    # If you use MSELoss, the loss is (prediction - target)^2.
    # If prediction is 0 and target is 6 (or vice versa), loss is 36.
    pred_mse = torch.tensor([0.0])
    target_mse = torch.tensor([6.0])
    
    # Default reduction is 'mean', but for a single item, mean = sum
    criterion_mse = nn.MSELoss() 
    loss_mse = criterion_mse(pred_mse, target_mse)
    
    print(f"Scenario 1: The 'Square of Six' (MSE)")
    print(f"Target: {target_mse.item()}, Prediction: {pred_mse.item()}")
    print(f"Calculation: ({target_mse.item()} - {pred_mse.item()})^2")
    print(f"Resulting Loss: {loss_mse.item()}")
    print("-" * 30)

    # SCENARIO 2: The "Sum Reduction" Trap
    # If you use reduction='sum', errors accumulate across the batch.
    # Example: Batch of 4 items, each with an error of 3.
    # (3^2 = 9). 9 * 4 items = 36.
    batch_size = 4
    pred_batch = torch.zeros(batch_size)
    target_batch = torch.full((batch_size,), 3.0) # All targets are 3.0
    
    criterion_sum = nn.MSELoss(reduction='sum') 
    loss_sum = criterion_sum(pred_batch, target_batch)

    print(f"Scenario 2: Sum Reduction (Accumulated Batch Error)")
    print(f"Batch Size: {batch_size}")
    print(f"Error per item: 3.0 (Squared error = 9.0)")
    print(f"Calculation: 9.0 * 4 items")
    print(f"Resulting Loss: {loss_sum.item()}")
    print("-" * 30)

    # SCENARIO 3: The 6x6 Grid (Common in Games/CNNs)
    # If you have a 6x6 output (36 pixels/cells) and every single one
    # has an error of 1.0, and you sum them up.
    grid_size = 6
    pred_grid = torch.zeros((grid_size, grid_size))
    target_grid = torch.ones((grid_size, grid_size)) # All ones
    
    criterion_grid = nn.MSELoss(reduction='sum')
    loss_grid = criterion_grid(pred_grid, target_grid)

    print(f"Scenario 3: The 6x6 Grid Sum")
    print(f"Grid Size: {grid_size}x{grid_size} = {grid_size*grid_size} elements")
    print(f"Error per element: 1.0")
    print(f"Calculation: Sum of 36 errors of 1.0")
    print(f"Resulting Loss: {loss_grid.item()}")
    print("-" * 30)

    # SCENARIO 4: L1 Loss (Absolute Difference)
    # Less likely to be exactly 36 unless your target is huge.
    pred_l1 = torch.tensor([0.0])
    target_l1 = torch.tensor([36.0])
    criterion_l1 = nn.L1Loss()
    loss_l1 = criterion_l1(pred_l1, target_l1)

    print(f"Scenario 4: L1 Loss (Absolute Error)")
    print(f"Target: {target_l1.item()}, Prediction: {pred_l1.item()}")
    print(f"Calculation: |{target_l1.item()} - {pred_l1.item()}|")
    print(f"Resulting Loss: {loss_l1.item()}")
    print("-" * 30)

if __name__ == "__main__":
    diagnose_loss_36()