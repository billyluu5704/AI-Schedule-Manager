# 🧠 Fuzzy Task Suitability Agent

This repository implements a **hybrid AI system** for task assignment that combines:

- **Task description generation** using a fine-tuned [LLaMA 3.1 8B Instruct](https://huggingface.co/meta-llama) model  
- **Availability estimation** using a fuzzy inference system  
- **Suitability scoring** that considers staff skills, experience, workload, and availability  

It is designed to support intelligent scheduling and staff allocation in research, academic, and engineering teams.

---

## 🚀 Features

- **Natural language task input** → The system parses free-text task descriptions into structured requirements (deadline, skills, complexity, priority, NTS skills (Non-technical skills), importance).
- **Fuzzy availability engine** → Evaluates how busy a staff member is, considering deadlines, workload pressure, and context switching.
- **Fuzzy suitability engine** → Produces a final *suitability score (0–100)* by combining:
  - Availability  
  - Skill match  
  - Experience level  
  - Current workload  
  - Task complexity & priority  
  - Non-technical skills average score
  - Importance average score
- **Fallback rules & safeguards** to avoid empty fuzzy outputs.

---

## 📂 Project Structure
    ```bash
    AI-Scheduling-Manager/
    │──data
        │── tech_company_detailed_tasks_timeAdjusted_v2_importanceFixed.csv
    │──notebooks
        │── fine_tune_instruction.ipynb
        │── task_describe_agent_notebook.ipynb
    │──pipeline
        │── task_describe_agent.py # LLM-based task parser
        │── fuzzy_logic_for_availability_check.py # Fuzzy availability engine
        │── fuzzy_system.py # Main entrypoint for suitability scoring
    │──plots
        │── instruction_loss_curve.png
    │──training
        │── fine_tune_instruction.py
    │── requirements.txt # Python dependencies
    │── README.md # This file
    ```


---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/billyluu5704/AI-Schedule-Manager.git
   cd AI-Schedule-Manager
   pip install -r requirements.txt
   ```

---

## 🧩 Example Input
    ```bash
    user_input = 
    """
    Design and implement a machine learning pipeline to predict customer churn.
    Use Python with scikit-learn or XGBoost, and ensure the model can be evaluated
    using ROC-AUC and F1 metrics. Collaborate with data engineering for dataset setup.
    Deadline: 2026-11-12 18:00
    """

    staff = {
        "availability": [
            {"deadline": "2025-12-15 17:00"},  
            {"deadline": "2026-02-01 17:00"}
        ],
        "skills": ["Python", "scikit-learn", "XGBoost", "ROC-AUC", "F1", "Airflow", "Data Engineering"],
        "experience": "junior",
        "workload": 2
    }
    ```

---

## 🧩 Expected Output
    ```bash
    {"Availability Score": 67}
    Suitability Score: 72.5
    Final Suitability Score: 72.5
    ```

---

## 📊 Fuzzy Rules (Highlights)

High availability + high skill → High suitability

Low availability or skill → Low suitability

Medium cases balanced by workload and experience

Junior staff supported with fallback rules
