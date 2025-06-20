from dotenv import load_dotenv
load_dotenv(override=True)

from .load_data import df_task
from .graph.graph import app

task = df_task["Task"][1]

state = app.invoke({"task": task})

print(state)

print(state['code'])