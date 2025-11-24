import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import time
import json
from huggingface_hub import login
login(token = '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')
from pipeline.task_describe_agent import generate
#from availability_check_agent import generate_score
from pipeline.fuzzy_logic_for_availability_check import fuzzy_availability
import os
os.makedirs(r"D:\hf_cache\hub", exist_ok=True)

# Best: point directly to the hub cache folder
#os.environ["HUGGINGFACE_HUB_CACHE"] = r"D:\hf_cache\hub"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/mnt/d/hf_cache/hub"

base_model = "meta-llama/Llama-3.2-3B-Instruct"

user_input = """Design and implement a machine learning pipeline to predict customer churn using historical transaction and interaction data. 
Use Python with scikit-learn or XGBoost, and ensure the model can be evaluated using ROC-AUC and F1 metrics. 
Collaborate with the data engineering team to source clean datasets and set up daily retraining jobs via Airflow. 
The project must be delivered by 2026-06-26 14:00."""


staff = {
    "availability": [
        {"deadline": "2025-12-15 17:00"},  # about 1 month away
        {"deadline": "2026-02-01 17:00"}
    ],
    "skills": [
        "Python", "scikit-learn", "XGBoost", 
        "ROC-AUC", "F1", "Airflow", "Data Engineering"
    ],
    "experience": "junior",       # gives access to your top-tier rules
}

def fuzzy_input(user_input, staff):
    # 1) Call the task description agent
    raw_task = generate(user_input)

    # 2) Normalize to a Python dict
    if isinstance(raw_task, str):
        # If generate() returns JSON string
        task = json.loads(raw_task)
    else:
        # If generate() already returns a dict
        task = raw_task

    # 3) Handle different possible key spellings for the deadline
    deadline = (
        task.get("Deadline")
        or task.get("deadline")
        or task.get("TaskDeadline")
    )
    if deadline is None:
        raise KeyError(f"No deadline field found in task. Got keys: {list(task.keys())}")

    # 4) Build availability payload for the fuzzy availability system
    user_availability = {
        "Staff Projects": staff["availability"],
        "New Task": {
            "deadline": deadline
        }
    }

    # 5) Run availability fuzzy system
    response = fuzzy_availability(json.dumps(user_availability))
    availability_score = json.loads(response)  # e.g. {"Availability Score": 87}

    print("Parsed task:", task)
    print("Availability Score:", availability_score)

    return task, availability_score

# Skill match function
def compute_skill_match(staff_skills, required_skills):
    staff_set = set(map(str.lower, staff_skills))
    required_set = set(map(str.lower, required_skills))
    return round(100 * len(staff_set & required_set) / len(required_set), 2)

def compute_nts_importance_avgs(task: dict):
    dims = [
        "ambiguity_tolerance",
        "communication",
        "planning",
        "collaboration",
        "reasoning",
        "risk_awareness",
        "ownership",
        "stakeholder_mgmt",
    ]

    nts = task.get("NTS_skills", {}) or {}
    imp = task.get("importance", {}) or {}

    # Default to 0 if a key is missing (you can change to 3 if you prefer neutral)
    nts_vals = [float(nts.get(k, 0)) for k in dims]
    imp_vals = [float(imp.get(k, 0)) for k in dims]

    nts_avg = sum(nts_vals) / len(dims)
    imp_avg = sum(imp_vals) / len(dims)

    return nts_avg, imp_avg


def fuzzy_system(user_input, staff):
    task, availability_score = fuzzy_input(user_input=user_input, staff=staff)
    nts_avg_val, importance_avg_val = compute_nts_importance_avgs(task)
    # Map task priority string to value
    priority_map = {"low": 0, "medium": 1, "high": 2}
    experience_map = {
        "entry_level": 0,
        "junior": 1,
        "mid_level": 2,
        "senior": 3
    }

    if availability_score is None:
        print("No valid availability score found.")
        return None
    
    # Define fuzzy variables
    availability = ctrl.Antecedent(np.arange(0, 101, 1), 'availability')
    skill_match = ctrl.Antecedent(np.arange(0, 101, 1), 'skill_match')
    experience = ctrl.Antecedent(np.arange(0, 5, 1), 'experience') #fix: depending on your staff's experience on each skill
    workload = ctrl.Antecedent(np.arange(0, 11, 1), 'workload') #number of tasks
    task_priority = ctrl.Antecedent(np.arange(0, 3, 1), 'task_priority') #1-3 scale
    task_complexity = ctrl.Antecedent(np.arange(1, 11, 1), 'task_complexity')
    nts_avg = ctrl.Antecedent(np.arange(1, 5.1, 0.1), 'nts_avg')
    importance_avg = ctrl.Antecedent(np.arange(1, 5.1, 0.1), 'importance_avg')

    #define fuzzy output variable
    suitability = ctrl.Consequent(np.arange(0, 101, 1), 'suitability')

    # Membership functions
    task_priority['low'] = fuzz.trimf(task_priority.universe, [0, 0, 1])
    task_priority['medium'] = fuzz.trimf(task_priority.universe, [0.5, 1, 1.5])
    task_priority['high'] = fuzz.trimf(task_priority.universe, [1, 2, 2])

    task_complexity['low'] = fuzz.trimf(task_complexity.universe, [1, 1, 4])
    task_complexity['medium'] = fuzz.trimf(task_complexity.universe, [3, 5, 8])
    task_complexity['high'] = fuzz.trimf(task_complexity.universe, [7, 10, 10])

    availability['low']    = fuzz.trapmf(availability.universe, [0, 0, 30, 45])
    availability['medium'] = fuzz.trimf(availability.universe,   [40, 55, 70])
    availability['high']   = fuzz.trapmf(availability.universe,  [65, 80, 92, 96])
    availability['very_high'] = fuzz.trapmf(availability.universe, [90, 94, 100, 100])

    skill_match['low'] = fuzz.trimf(skill_match.universe, [0, 0, 50])
    skill_match['medium'] = fuzz.trimf(skill_match.universe, [30, 50, 70])
    skill_match['high'] = fuzz.trimf(skill_match.universe, [60, 100, 100])

    experience['entry_level'] = fuzz.trimf(experience.universe, [0, 0, 1])
    experience['junior'] = fuzz.trimf(experience.universe, [0, 1, 2])
    experience['mid_level'] = fuzz.trimf(experience.universe, [1, 2, 3])
    experience['senior'] = fuzz.trimf(experience.universe, [2, 3, 4])

    workload['light'] = fuzz.trimf(workload.universe, [0, 0, 4])
    workload['moderate'] = fuzz.trimf(workload.universe, [3, 5.5, 8])
    workload['heavy'] = fuzz.trimf(workload.universe, [7, 10, 10])

    # NTS_skills average (1–5)
    nts_avg['low']    = fuzz.trimf(nts_avg.universe, [1.0, 1.0, 3.0])
    nts_avg['medium'] = fuzz.trimf(nts_avg.universe, [2.5, 3.0, 3.5])
    nts_avg['high']   = fuzz.trimf(nts_avg.universe, [3.0, 5.0, 5.0])

    # importance average (1–5)
    importance_avg['low']    = fuzz.trimf(importance_avg.universe, [1.0, 1.0, 3.0])
    importance_avg['medium'] = fuzz.trimf(importance_avg.universe, [2.5, 3.0, 3.5])
    importance_avg['high']   = fuzz.trimf(importance_avg.universe, [3.0, 5.0, 5.0])

    suitability['low']    = fuzz.trapmf(suitability.universe, [0, 0, 30, 45])
    suitability['medium'] = fuzz.trimf(suitability.universe, [40, 55, 70])
    suitability['high']   = fuzz.trapmf(suitability.universe, [70, 85, 100, 100])

    #define fuzzy rules
    rules = [

        # 🟥 Critical tasks → Only top performers
        ctrl.Rule(
            task_priority['high'] & task_complexity['high'] &
            availability['high'] & skill_match['high'] &
            experience['senior'] & workload['light'],
            suitability['high']
        ),

        # 🟧 High priority, medium complexity → allow high skill & good availability
        ctrl.Rule(
            task_priority['high'] & task_complexity['medium'] &
            skill_match['high'] & (experience['senior'] | experience['mid_level']) &
            (availability['high'] | availability['medium']) &
            (workload['light'] | workload['moderate']),
            suitability['high']
        ),

        # 🟨 Medium priority, high complexity → allow strong skill or experience
        ctrl.Rule(
            task_priority['medium'] & task_complexity['high'] &
            (skill_match['high'] | experience['senior']) &
            (availability['medium'] | availability['high']) &
            workload['moderate'],
            suitability['medium']
        ),

        # 🟩 Medium priority, medium complexity → ideal for mid-levels
        ctrl.Rule(
            task_priority['medium'] & task_complexity['medium'] &
            skill_match['medium'] & experience['mid_level'] &
            availability['medium'] & workload['moderate'],
            suitability['medium']
        ),

        # 🟦 Low priority, low complexity → suitable for entry/junior/mid with light workload
        ctrl.Rule(
            task_priority['low'] & task_complexity['low'] &
            (experience['entry_level'] | experience['junior'] | experience['mid_level']) &
            (skill_match['medium'] | skill_match['high']) &
            workload['light'],
            suitability['medium']
        ),

        # ⛔ Any of these → strongly lower suitability
        ctrl.Rule(
            skill_match['low'] | workload['heavy'] | availability['low'],
            suitability['low']
        ),

        # ⚠️ Entry-level with low availability and low skill → very poor match
        ctrl.Rule(
            availability['low'] & skill_match['low'] & experience['entry_level'],
            suitability['low']
        )
    ]

    # --- keep your existing rules, then ADD these ---
    rules += [
        # Broad “good match” when availability & skill are strong, regardless of title
        ctrl.Rule((availability['high'] & skill_match['high']) &
                (workload['light'] | workload['moderate']),
                suitability['high']),

        # Solid medium when everything is ... medium-ish
        ctrl.Rule((availability['medium'] & skill_match['medium']) &
                (workload['light'] | workload['moderate']),
                suitability['medium']),

        # Junior case with decent availability & low workload
        ctrl.Rule((experience['junior'] & availability['high'] &
                (skill_match['medium'] | skill_match['high']) &
                workload['light']),
                suitability['medium']),
    ]

    # Strengthen the obvious good-case path:
    rules += [
        ctrl.Rule(
            (availability['high'] | availability['very_high']) &
            (skill_match['high']) &
            (workload['light'] | workload['moderate']),
            suitability['high']
        ),
    ]

        # ✅ High NTS + high importance → boost suitability
    rules += [
        ctrl.Rule(
            nts_avg['high'] & importance_avg['high'] &
            (availability['high'] | availability['very_high']) &
            (skill_match['medium'] | skill_match['high']),
            suitability['high']
        ),

        # ⚠ Medium NTS/importance → keep in medium band if other factors are not extreme
        ctrl.Rule(
            nts_avg['medium'] & importance_avg['medium'] &
            (availability['medium'] | availability['high']) &
            (skill_match['medium'] | skill_match['high']),
            suitability['medium']
        ),

        # ⛔ Low NTS or low importance → drag suitability down
        ctrl.Rule(
            (nts_avg['low'] | importance_avg['low']) &
            (task_priority['high'] | task_complexity['high']),
            suitability['low']
        ),
    ]

    # --- Good-case ⇒ high (dominant) ---
    r_good = ctrl.Rule(
        (availability['high'] | availability['very_high']) &
        (skill_match['high']) &
        (workload['light'] | workload['moderate']),
        suitability['high']
    )
    r_good.weight = 1.0   # max influence

    # --- High priority + small/medium complexity ⇒ high ---
    r_hp_sm = ctrl.Rule(
        task_priority['high'] &
        (task_complexity['low'] | task_complexity['medium']) &
        (availability['high'] | availability['very_high']) &
        (skill_match['high']) &
        (workload['light'] | workload['moderate']),
        suitability['high']
    )
    r_hp_sm.weight = 0.95

    # --- Narrow safety-net medium (replaces the broad catch-all) ---
    r_medium_net = ctrl.Rule(
        (availability['medium']) &
        (skill_match['medium']) &
        (workload['light'] | workload['moderate']) &
        ~(task_priority['high'] & task_complexity['high']),
        suitability['medium']
    )
    r_medium_net.weight = 0.6  # keep it from overpowering the high case

    # (Optional) Senior bonus in strong cases
    r_senior = ctrl.Rule(
        (experience['senior']) &
        (availability['high'] | availability['very_high']) &
        (skill_match['high']) &
        (workload['light']),
        suitability['high']
    )
    r_senior.weight = 0.9
    rules += [r_good, r_hp_sm, r_medium_net, r_senior]

    # Replace the broad catch-all with this narrower rule:
    r_medium_net = ctrl.Rule(
        (availability['medium']) &
        (skill_match['medium']) &
        (workload['light'] | workload['moderate']) &
        ~(task_priority['high'] & task_complexity['high']),
        suitability['medium']
    )
    r_medium_net.weight = 0.6  # light influence so it won't dominate
    rules += [r_medium_net]

    # Try both keys, fall back to empty list / string if neither exists
    required_skills = task.get('Required Skills')
    if required_skills is None:
        required_skills = task.get('RequiredSkills', [])

    # Create control system and simulation
    suit_ctrl = ctrl.ControlSystem(rules)
    suitability_engine = ctrl.ControlSystemSimulation(suit_ctrl)

    suitability_engine.input['availability']   = float(np.clip(availability_score["Availability Score"], 0, 100))
    suitability_engine.input['skill_match']    = float(np.clip(compute_skill_match(staff['skills'], required_skills), 0, 100))
    suitability_engine.input['experience']     = float(np.clip(experience_map[staff['experience']], 0, 4))
    current_workload = len(staff.get('availability', [])) #number of projects
    suitability_engine.input['workload']       = float(np.clip(current_workload, 0, 10))
    suitability_engine.input['task_priority']  = float(np.clip(priority_map[task['Task Priority']], 0, 2))
    suitability_engine.input['task_complexity']= float(np.clip(task['Task Complexity'], 1, 10))
    suitability_engine.input['nts_avg']        = float(np.clip(nts_avg_val, 1, 5))
    suitability_engine.input['importance_avg'] = float(np.clip(importance_avg_val, 1, 5))


    suitability_engine.compute()
    suit = suitability_engine.output.get('suitability')
    if suit is None or np.isnan(suit):
        suit = 50.0  # conservative default
        
    suitability_score = round(float(suit), 2)
    print("Suitability Score:", suitability_score)
    return suitability_score

if __name__ == "__main__":
    suitability_score = fuzzy_system(user_input, staff)
    if suitability_score is not None:
        print(f"Final Suitability Score: {suitability_score}")
    else:
        print("Failed to compute suitability score.")


#To run:
#conda activate my_env
#python fuzzy_system.py