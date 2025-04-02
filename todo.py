# Install necessary packages (Colab or local)
!pip install -q transformers python-dateutil ics
!pip install pytz

import datetime
import re
import json
import os
from dateutil import parser as dateparser
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from ics import Calendar, Event
import pytz

# Timezone setup
LOCAL_TZ = pytz.timezone("America/Toronto")

# Load FLAN-T5 model
model_name = "google/flan-t5-xl"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Task memory
tasks = []
DATA_FILE = "tasks.json"

# Color codes for categories
COLOR_CODES = {
    "school": "\033[94m",  # Blue
    "work": "\033[92m",    # Green
    "life": "\033[93m",    # Yellow
    "end": "\033[0m"
}

def ai_response(prompt):
    task_summary = "\n".join([
        f"- {t['name']} (Due: {t['due'].strftime('%Y-%m-%d %H:%M')}, Priority: {t['priority']}, Category: {t.get('category', 'life')})"
        for t in tasks if not t["done"]
    ]) or "No active tasks."

    full_prompt = (
        "You are a helpful productivity assistant. Use the task list below to give an accurate and thoughtful answer.\n\n"
        f"Tasks:\n{task_summary}\n\n"
        f"User: {prompt}\nAssistant:"
    )

    result = generator(full_prompt, max_length=256, do_sample=True, temperature=0.7)
    return result[0]['generated_text']

def normalize(text):
    return re.sub(r"\W+", "", text.lower())

def save_tasks():
    with open(DATA_FILE, "w") as f:
        json.dump([
            {
                "name": t["name"],
                "due": t["due"].strftime("%Y-%m-%d %H:%M"),
                "priority": t["priority"],
                "repeat": t["repeat"],
                "done": t["done"],
                "category": t.get("category", "life")
            } for t in tasks
        ], f)

def load_tasks():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            loaded = json.load(f)
            for t in loaded:
                t["due"] = datetime.datetime.strptime(t["due"], "%Y-%m-%d %H:%M")
                t["category"] = t.get("category", "life")
                tasks.append(t)

load_tasks()

def parse_task(user_input):
    parts = [p.strip() for p in user_input.split(",")]
    task = parts[0]
    priority = "normal"
    category = "life"
    repeat = "none"
    due = datetime.datetime.now() + datetime.timedelta(hours=1)

    if user_input.strip().endswith("?"):
        return None

    if len(parts) > 1 and parts[1].lower() in ["high", "normal", "low"]:
        priority = parts[1].lower()
    if len(parts) > 2 and parts[2].lower() in ["school", "work", "life"]:
        category = parts[2].lower()

    match = re.search(r"every (\w+)", task.lower())
    weekdays = {
        "monday": 0, "tuesday": 1, "wednesday": 2,
        "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6
    }

    if match:
        repeat = "weekly"
        weekday = match.group(1).lower()
        if weekday in weekdays:
            today = datetime.datetime.now()
            days_ahead = (weekdays[weekday] - today.weekday()) % 7
            next_occurrence = today + datetime.timedelta(days=days_ahead)

            time_match = re.search(r"at (\d{1,2})(?::(\d{2}))?\s*(am|pm)?", task.lower())
            hour, minute = 9, 0
            if time_match:
                hour = int(time_match.group(1))
                minute = int(time_match.group(2)) if time_match.group(2) else 0
                ampm = time_match.group(3)
                if ampm == "pm" and hour != 12:
                    hour += 12
                elif ampm == "am" and hour == 12:
                    hour = 0
            due = next_occurrence.replace(hour=hour, minute=minute)
    else:
        try:
            due = dateparser.parse(task, fuzzy=True)
        except:
            pass

    return task, due, priority, repeat, category

def add_task(name, due, priority, repeat, category):
    tasks.append({
        "name": name,
        "due": due,
        "priority": priority,
        "repeat": repeat,
        "done": False,
        "category": category
    })
    save_tasks()
    print(f"Pending Task: '{name}' added for {due.strftime('%Y-%m-%d %H:%M')} in category [{category}]")

def complete_task(task):
    if task["repeat"] == "none":
        tasks.remove(task)
        print(f"Deleted '{task['name']}' (non-repeating task).")
    else:
        task["done"] = True
        print(f"Completed Task: '{task['name']}' marked as done.")
    save_tasks()

def clear_all_tasks():
    confirm = input("Are you sure you want to delete all tasks? (yes/no): ").strip().lower()
    if confirm == "yes":
        tasks.clear()
        if os.path.exists(DATA_FILE):
            os.remove(DATA_FILE)
        print("All tasks have been cleared.")
    else:
        print("Clear operation canceled.")

def show_week():
    print("\nWeek Overview:")
    today = datetime.datetime.now()
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    for i in range(7):
        day_index = (today.weekday() + i) % 7
        day_name = weekdays[day_index]
        print(f"\nTasks for {day_name}:")
        found = False
        current_day = today + datetime.timedelta(days=i)
        current_day = current_day.replace(hour=0, minute=0, second=0, microsecond=0)
        for t in tasks:
            due = t["due"]
            adjusted_due = due
            if t["repeat"] == "weekly":
                if due.weekday() != current_day.weekday():
                    continue
                adjusted_due = current_day.replace(hour=due.hour, minute=due.minute)
            if adjusted_due.date() == current_day.date():
                found = True
                status = "COMPLETED" if t["done"] else "PENDING"
                color = COLOR_CODES.get(t.get("category", "life"), "")
                reset = COLOR_CODES["end"]
                print(f"  - {color}{t['name']}{reset} | Due: {adjusted_due.strftime('%Y-%m-%d %H:%M')} | Priority: {t['priority']} | {status}")
        if not found:
            print("  No tasks for that day!")

def show_all_tasks():
    print("\nAll Tasks:")
    if not tasks:
        print("No tasks added yet.")
        return
    for i, t in enumerate(tasks):
        status = "COMPLETED" if t["done"] else "PENDING"
        color = COLOR_CODES.get(t.get("category", "life"), "")
        reset = COLOR_CODES["end"]
        print(f"{i+1}. {color}{t['name']}{reset} | Due: {t['due'].strftime('%Y-%m-%d %H:%M')} | Priority: {t['priority']} | Repeat: {t['repeat']} | Category: {t.get('category', 'life')} | {status}")

# Main loop
print("\nSmart Task Assistant: Now with AI support!")
print("Type tasks like: 'Remind me to hand in my Capstone Video on Monday at 11:59pm, high, school'")
print("Commands: show week | show all | organize week | export calendar | clear all | ai [question] | done | exit\n")

while True:
    user_input = input("What can I help you with? ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    elif user_input.lower().startswith("ai "):
        prompt = user_input[3:].strip()
        print(ai_response(prompt))
        continue

    elif user_input.strip().endswith("?"):
        print(ai_response(user_input.strip()))
        continue

    elif user_input.lower() == "clear all":
        clear_all_tasks()

    elif user_input.lower() == "show week":
        show_week()

    elif user_input.lower() == "show all":
        show_all_tasks()

    elif user_input.lower() == "done":
        show_all_tasks()
        try:
            num = int(input("Which task number to mark as done? ")) - 1
            if 0 <= num < len(tasks):
                complete_task(tasks[num])
            else:
                print("Invalid number.")
        except:
            print("Invalid input.")

    else:
        result = parse_task(user_input)
        if result:
            task, due, priority, repeat, category = result
            add_task(task, due, priority, repeat, category)
        else:
            print("Couldn't understand. Try rephrasing or use 'ai [question]'.")
