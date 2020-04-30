import tensorflow as tf 
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import json
import os


def tfevents_to_json(path):
    path = '.'
    events = [filename for filename in os.listdir(path) if filename.startswith("events.out")]
    for event in events:
        event_acc = EventAccumulator(event).Reload()
        logs = dict()
        for scalar_key in event_acc.scalars.Keys():
            scalar_dict = {"wall_time": [], "step": [], "value": []}
            for scalars in event_acc.Scalars(scalar_key):
                scalar_dict["wall_time"].append(scalars.wall_time)
                scalar_dict["step"].append(scalars.step)
                scalar_dict["value"].append(scalars.value)
            logs[scalar_key] = scalar_dict

        with open("logs.json", "w") as f:
            json.dump(logs, f)

