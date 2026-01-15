# Beyond the Move 

> *Felix qui potuit rerum cognoscere causas.* > "Happy is he who is able to know the causes of things." — Virgil
>
> *初心忘るべからず (Shoshin wasuru bekarazu)* > "Never forget the beginner's spirit." — Zeami Motokiyo

[![Project Status: Phase I Completed](https://img.shields.io/badge/Phase%20I-Completed-success)](https://github.com/4idoneus)
[![Framework](https://img.shields.io/badge/PyTorch-ResNet18-red)](https://pytorch.org/)
[![XAI](https://img.shields.io/badge/XAI-Integrated%20Gradients-blue)](https://captum.ai/)
[![License](https://img.shields.io/badge/License-Academic-lightgrey)]()

##  Abstract

**Current AI engines are Oracles, not Tutors.**
While Deep Reinforcement Learning (DRL) agents like AlphaGo have achieved superhuman performance, they operate as "Black Boxes." They provide optimal moves but offer minimal transparency regarding their strategic rationale. This creates an **"Epistemic Gap"** where human learners can observe *what* the AI plays, but cannot understand *why*.

**Beyond the Move** is an Explainable AI (XAI) framework designed to bridge this gap. This project dismantles the barrier between silicon calculation and human understanding, translating the silent probability of a neural network into the articulate language of strategy. By visualising **"Interaction Primitives"** via heatmaps, this system acts as a "Glass Box" tool for cognitive skill acquisition in the game of Go.

##  Project Scope & Architecture

This repository hosts the source code for **Phase I (Graduation Project I)**, creating a fully functional end-to-end pipeline from raw SGF data to visual explanation.

### Key Features (Phase I Delivered)
* ** The Brain (ResNet-18):** A Deep Residual Network modified to accept a $19 \times 19 \times 17$ input tensor, trained on 150,000+ professional games to predict moves with >80% Top-5 Accuracy.
* ** The Light (XAI):** Implementation of **Integrated Gradients (IG)** using the Captum library to visualize pixel-level attribution.
* ** Robust Data Pipeline:** A custom "Universal SGF Parser" capable of handling multi-encoding (UTF-8/GB18030) game records without data loss.
* ** Interactive UI:** A reactive web application built with **Streamlit** for real-time board analysis.
* ** Computer Vision (Experimental):** A WIP module using OpenCV for digitising physical boards via homography and adaptive thresholding.

##  Roadmap

### ✅ Phase I: Baseline Architecture (Jan 2026)
* Completed data ingestion of the FoxGo Dataset (150k+ games).
* Trained ResNet-18 baseline, achieving professional move alignment.
* Integrated XAI layer (Integrated Gradients) for "Where" visualisation.
* Deployed Streamlit frontend for user interaction.

###  Phase II: Semantic Explainability (Feb - June 2026)
* **Transition to Semantics:** Integration of **Grad-CAM** and **Saliency Maps** to visualise high-level shapes (e.g., "Tiger's Mouth") rather than just pixels.
* **Topological Learning:** Experimentation with **Graph Neural Networks (GNNs)** to capture stone connectivity and "Aji".
* **Benchmarking:** A comparative user study to validate XAI fidelity against professional intuition.

### Phase III: Future Research (2026+)
* Alignment with **COST Action CA22145 (GameTable)** research goals.
* Advanced cognitive modelling for human-AI collaboration.

##  Tech Stack

* **Language:** Python 3.10+
* **Deep Learning:** PyTorch, TorchVision
* **XAI Library:** Captum (Integrated Gradients)
* **Frontend:** Streamlit
* **Computer Vision:** OpenCV (cv2)
* **Data Processing:** NumPy, Pandas

##  About the Author

**İpek Naz Sipahi (Aidoneus)**
*B.E. Candidate, Computer Engineering | Manisa Celal Bayar University*

My academic focus lies in the "Main Quest" of mastering Cognitive Game AI and bridging the gap between machine logic and human intuition. This project serves as my senior graduation thesis and a foundation for future postgraduate research in Japan.

* **Advisor:** Dr Gamze TÜRKMEN 
* **Research Group:** COST Action CA22145 (GameTable) - Working Group 1 Member

##  Citation & License

This work is part of an ongoing academic thesis.
* **Code:** Open for academic review.
* **Dataset:** Utilizes the FoxGo Dataset (GPL-3.0).
* **Rights:** All rights reserved pending the final submission of the graduation project.

---
*"To play a stone is to ask a question. To analyse it is to understand the answer."*
