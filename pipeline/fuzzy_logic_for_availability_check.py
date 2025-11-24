import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import json, datetime as dt

def business_days_until(due: dt.datetime, now=None):
    if now is None:
        now = dt.datetime.now()
    days = np.busday_count(now.date(), due.date())
    # If due is later today (same date), count as 1 business day
    if days == 0 and due.date() == now.date() and due > now:
        return 1
    return int(days)


def parse_payload(user_input: str):
    # Accept dicts or JSON strings
    if isinstance(user_input, (str, bytes)):
        d = json.loads(user_input)
    else:
        d = user_input
        
    # Support both naming schemes
    projects = (
        d.get("projects")
        if "projects" in d
        else d.get("Staff Projects", [])
    )
    new_task = (
        d.get("new_task")
        if "new_task" in d
        else d["New Task"]
    )

    now = dt.datetime.now()
    new_deadline = dt.datetime.strptime(new_task["deadline"], "%Y-%m-%d %H:%M")
    due = business_days_until(new_deadline, now=now)
    past_due_flag = new_deadline <= now  # treat current date as past

    # --- Workload + overlap only from deadlines ---
    S = 0                      # count of tasks contributing to pressure
    nearest_deadline = None
    overlap_count = 0
    K = 3                      # overlap window in days
    future_deadlines = []

    for p in projects:
        past_deadline = dt.datetime.strptime(p["deadline"], "%Y-%m-%d %H:%M")

        # count any task due on/before the new task as one "unit" of workload
        if past_deadline <= new_deadline:
            S += 1

        # collect future deadlines to compute buffer
        if past_deadline >= now:
            future_deadlines.append(past_deadline)

        # count overlapping tasks (within ±K days)
        if abs((past_deadline.date() - new_deadline.date()).days) <= K:
            overlap_count += 1

    # workload_pressure = tasks per business day until new task
    workload_pressure = (S / max(1, due)) if due > 0 else 2.5  # saturate if past/zero

    # use nearest future deadline for buffer; if none, treat as large buffer
    if future_deadlines:
        nearest_deadline = min(future_deadlines)
        deadline_buffer_days = business_days_until(nearest_deadline, now=now)
    else:
        deadline_buffer_days = 30.0  # generous buffer when nothing coming up

    overlap_severity = (overlap_count / max(1, len(projects))) if projects else 0.0
    context_switch_load = len(projects)

    return dict(
        workload_pressure=float(np.clip(workload_pressure, 0, 2.5)),
        deadline_buffer_days=float(np.clip(deadline_buffer_days, 0, 30)),
        overlap_severity=float(np.clip(overlap_severity, 0, 1)),
        context_switch_load=float(np.clip(context_switch_load, 0, 12)),
        _past_due=bool(past_due_flag)
    )


def fuzzy_system_availability():
    work_pressure = ctrl.Antecedent(np.linspace(0, 2.5, 251), 'workload_pressure')
    buffer       = ctrl.Antecedent(np.linspace(0, 30, 301), 'deadline_buffer_days')
    overlap      = ctrl.Antecedent(np.linspace(0, 1, 101),  'overlap_severity')
    context      = ctrl.Antecedent(np.linspace(0, 12, 121), 'context_switch_load')

    availability = ctrl.Consequent(np.linspace(0, 100, 101), 'availability')

    # Work pressure
    work_pressure['low']  = fuzz.trapmf(work_pressure.universe, [0.0, 0.0, 0.45, 0.7])
    work_pressure['med']  = fuzz.trimf(work_pressure.universe, [0.4, 0.9, 1.2])
    work_pressure['high'] = fuzz.trapmf(work_pressure.universe, [1.0, 1.2, 2.5, 2.5])

    # Buffer (0..30)
    buffer['small'] = fuzz.trapmf(buffer.universe, [0, 0, 2, 6])
    buffer['med']   = fuzz.trimf(buffer.universe, [4, 10, 16])
    buffer['large'] = fuzz.trapmf(buffer.universe, [12, 18, 30, 30])

    # Overlap
    overlap['low']  = fuzz.trapmf(overlap.universe, [0, 0, 0.2, 0.35])
    overlap['med']  = fuzz.trimf(overlap.universe, [0.25, 0.5, 0.75])
    overlap['high'] = fuzz.trapmf(overlap.universe, [0.6, 0.75, 1, 1])

    # Context switches
    context['few']      = fuzz.trapmf(context.universe, [0, 0, 1, 3])
    context['moderate'] = fuzz.trimf(context.universe, [2, 4, 6])
    context['many']     = fuzz.trapmf(context.universe, [5, 7, 12, 12])

    # Availability output
    availability['none']      = fuzz.trapmf(availability.universe, [0, 0, 10, 20])
    availability['low']       = fuzz.trimf(availability.universe, [15, 30, 45])
    availability['moderate']  = fuzz.trimf(availability.universe, [40, 55, 70])
    availability['high']      = fuzz.trapmf(availability.universe, [65, 78, 90, 95])
    availability['very_high'] = fuzz.trapmf(availability.universe, [88, 93, 100, 100])

    rules = [
        # High pressure or tiny buffer → almost no availability
        ctrl.Rule((work_pressure['high'] | buffer['small']), availability['none']),

        # Medium pressure + high overlap → low availability
        ctrl.Rule(work_pressure['med'] & overlap['high'], availability['low']),

        # Many projects + high overlap → low
        ctrl.Rule(context['many'] & overlap['high'], availability['low']),

        # Low pressure + large buffer + low overlap → very high
        ctrl.Rule(work_pressure['low'] & buffer['large'] & overlap['low'],
                  availability['very_high']),

        # Low pressure + medium buffer + low overlap + not too many projects → high
        ctrl.Rule(work_pressure['low'] & buffer['med'] & overlap['low'] &
                  (context['few'] | context['moderate']),
                  availability['high']),

        # Medium pressure + low overlap + decent buffer + reasonable context → high
        ctrl.Rule(work_pressure['med'] & overlap['low'] &
                  (buffer['med'] | buffer['large']) &
                  (context['few'] | context['moderate']),
                  availability['high']),

        # Medium overlap + low/med pressure + decent buffer → moderate
        ctrl.Rule(overlap['med'] &
                  (work_pressure['low'] | work_pressure['med']) &
                  (buffer['med'] | buffer['large']),
                  availability['moderate']),

        # Medium overlap but few/moderate contexts → moderate
        ctrl.Rule(overlap['med'] & (context['few'] | context['moderate']),
                  availability['moderate']),

        # Very low everything → very high availability
        ctrl.Rule(context['few'] & work_pressure['low'] & overlap['low'],
                  availability['very_high']),
    ]

    system = ctrl.ControlSystem(rules)
    sim = ctrl.ControlSystemSimulation(system)
    return sim



def fuzzy_availability(user_input_json: str):
    sim = fuzzy_system_availability()
    feats = parse_payload(user_input_json)
    print("Parsed features:", feats)

    # Hard rule: if the new task deadline is in the past, availability = 0
    if feats.pop('_past_due', False):
        return json.dumps({"Availability Score": 0})
    
    # If there's truly no current work, clamp to 100
    if (feats['context_switch_load'] == 0 and
        feats['workload_pressure'] == 0 and
        feats['overlap_severity'] == 0):
        return json.dumps({"Availability Score": 100})
    
    for k, v in feats.items():
        sim.input[k] = v

    try:
        sim.compute()
        score = float(sim.output['availability'])
        return json.dumps({"Availability Score": int(round(score))})
    except Exception:
        # Fallback if defuzzification fails for any reason
        return json.dumps({"Availability Score": 0})


""" print(fuzzy_availability(json.dumps({
    "Staff Projects": [
        {"deadline": "2025-12-15 17:00"},  # about 1 month away
        {"deadline": "2026-02-01 17:00"}   # a few months away
    ],
    "New Task": {
        "deadline": "2026-09-12 17:00"     # far in the future
    }
})))

# Case 2: heavy workload, overlapping near-term deadlines → LOW availability
print(fuzzy_availability(json.dumps({
    "Staff Projects": [
        {"name": "ROC" ,"deadline": "2025-11-20 12:00"},  # 3 days from now
        {"name": "ROC","deadline": "2025-12-21 17:00"},  # 4 days from now
        {"name": "ROC", "deadline": "2026-11-22 10:00"}   # 5 days from now
    ],
    "New Task": {
        "name": "ROC","deadline": "2025-11-21 18:00"     # squeezed into the same week
    }
}))) """
