Beyond The Move: Phase 1 Final Technical Roadmap
================================================

📅 Week 2: The Brain & Analytics (Dec 10 -- Dec 14)
--------------------------------------------------

**Goal:** Train the Neural Network, track data usage, and generate performance graphs.

### 📝 File 1: `src/dataset.py` (The Data Waiter)

**Task:** Create a robust loader that counts your data.

-   **To-Do:**

    -   Initialize with `glob` to find all `.npz` files in `data/processed`.

    -   **Data Counting:** Add a print statement: `print(f"Dataset Size: {len(self.files)} games loaded.")`.

    -   **Get Item:** Load `.npz`, extract `input` (19x19x17) and `target`. Convert to `torch.float32` and `torch.long`.

### 📝 File 2: `src/model.py` (The Architecture)

**Task:** Define the ResNet that acts as the "Intuition Engine."

-   **To-Do:**

    -   Import `resnet18`. Change `conv1` to 17 channels. Change `fc` to 361 classes.

    -   **The Formula:** Apply `F.log_softmax(x, dim=1)` at the end. This calculates the probability $P(move|board)$.

### 📝 File 3: `src/train.py` (The Gym)

**Task:** Train the model and generate the **Cost Function Diagram** for Page 3.

-   **To-Do:**

    -   **Setup:** Lists for logging `train_losses = []`.

    -   **Loop:** Checkpoint saving logic (save `model_epoch_X.pth` every epoch).

    -   **Diagram Generation:**

        ```
        import matplotlib.pyplot as plt
        plt.plot(train_losses)
        plt.title("Training Cost Function (Convergence)")
        plt.xlabel("Steps")
        plt.ylabel("Cross Entropy Loss")
        plt.savefig("assets/training_graph.png") # <--- SAVES DIAGRAM FOR PAGE 3

        ```

    -   **Metadata Saving:** Save a `training_stats.json` file containing: `{"total_games": 24500, "final_acc": 0.42}`.

📅 Week 3: The Interface & "The Teacher" (Dec 15 -- Dec 21)
----------------------------------------------------------

**Goal:** Replace hardcoded scenarios with live AI predictions and adaptive explanations.

### 📝 File 4: `src/predict.py` (The Strategist)

**Task:** Handle the logic for "My Move" and "Opponent's Reply."

-   **To-Do:**

    -   **Top Moves:** Return the top 5 moves and their confidence %.

    -   **Rival Prediction:**

        1.  Clone the board.

        2.  Play the AI's best move (Color A).

        3.  Run inference again for Color B.

        4.  Return that move as "Rival's Likely Response."

### 📝 File 5: `src/heuristics.py` (The Linguist)

**Task:** Generate text based on the 3 Levels (Novice/Int/Expert).

-   **To-Do:**

    -   **Geometry:** Identify Corner vs. Side vs. Center.

    -   **Influence:** Check local stone density (is it crowded?).

    -   **Templates:**

        -   `if mode == "Novice": return "Play here to make a safe base."`

        -   `if mode == "Expert": return "This move expands influence while reducing the opponent's thickness."`

### 📝 File 6: `app.py` (The Dashboard)

**Task:** Update the Streamlit UI.

-   **Page 2 (Interactive):**

    -   **Switch:** `color = st.radio("I am playing:", ["Black", "White"])`.

    -   **Input:** Accept SGF upload OR Image upload.

    -   **Display:** Show the board with Ghost Stones (AI suggestions).

    -   **Sidebar:** Show the heuristic text explanations.

-   **Page 3 (Analytics):**

    -   **Load:** Read `training_stats.json`.

    -   **Metric:** `st.metric("Total Professional Games Analyzed", stats['total_games'])`.

    -   **Visual:** `st.image("assets/training_graph.png")`.

📅 Week 4: The Soul (XAI) (Dec 22 -- Dec 28)
-------------------------------------------

**Goal:** Generate the "Why" (Heatmaps).

### 📝 File 7: `src/xai.py` (The Explainer)

**Task:** Use Integrated Gradients to show stone importance.

-   **To-Do:**

    -   Run `Captum.IntegratedGradients`.

    -   **Summing:** Collapse the 17 planes into 1 heatmap.

    -   **Overlay:** In `app.py`, draw Green squares over positive stones and Red squares over negative stones.

📅 Week 5: The Eyes (Vision) (Dec 29 -- Jan 4)
---------------------------------------------

**Goal:** Allow Real-World Photos.

### 📝 File 8: `src/vision.py` (The Observer)

**Task:** Convert a messy photo into a clean 19x19 matrix.

-   **To-Do:**

    -   **Grid:** Use `cv2.findContours` and `warpPerspective` to flatten the board.

    -   **Stones:** Use `cv2.HoughCircles` to find stones.

    -   **Classify:** Bright pixels = White, Dark pixels = Black.

    -   **Output:** Return a numpy array compatible with `model.py`.

📅 Week 6: Delivery (Jan 5 -- Jan 12)
------------------------------------

**Goal:** Documentation and Submission.

-   **Demo Video:** Record the full flow (Upload Photo -> AI Predicts -> Change Rank to Expert -> See Explanation).

-   **Final Report:** Include the Training Graph and XAI Heatmaps as "Results."