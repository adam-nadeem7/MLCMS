import json
import numpy as np
import subprocess
import random

# Specify the path to your JSON file
file_path = 'scenarios/SIR.scenario'

# Read the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)

true = True
false = False
null = None

x_position = 13.6
y_position = 2.1

scenario_file = "scenarios/SIR.scenario"
output_dir = "console_output"

pedestrians = []

target_idss = []

for i in range(0, len(data['scenario']['topography']['targets'])):
    target_id = data['scenario']['topography']['targets'][i]['id']
    target_idss.append(target_id)

# Loop to add 400 pedestrians
for i in range(1000):
    dynamicElement = {
        "attributes": {
            "id": i,  # Unique ID for each pedestrian
            "shape": {
                "x": 0.0,
                "y": 0.0,
                "width": 1.0,
                "height": 1.0,
                "type": "RECTANGLE"
            },
            "visible": True,
            "radius": 0.2,
            "densityDependentSpeed": False,
            "speedDistributionMean": 1.34,
            "speedDistributionStandardDeviation": 0.26,
            "minimumSpeed": 0.5,
            "maximumSpeed": 2.2,
            "acceleration": 2.0,
            "footstepHistorySize": 4,
            "searchRadius": 1.0,
            "walkingDirectionSameIfAngleLessOrEqual": 45.0,
            "walkingDirectionCalculation": "BY_TARGET_CENTER"
        },
        "source": None,
        "targetIds": target_idss,
        "nextTargetListIndex": 0,
        "isCurrentTargetAnAgent": False,
        "position": {
            "x": random.randint(1, 34),
            "y": random.randint(1, 34)
        },
        "velocity": {
            "x": 0.0,
            "y": 0.0
        },
        "freeFlowSpeed": 1.3653335527746429,
        "followers": [],
        "idAsTarget": -1,
        "isChild": False,
        "isLikelyInjured": False,
        "psychologyStatus": {
            "mostImportantStimulus": None,
            "threatMemory": {
                "allThreats": [],
                "latestThreatUnhandled": False
            },
            "selfCategory": "TARGET_ORIENTED",
            "groupMembership": "OUT_GROUP",
            "knowledgeBase": {
                "knowledge": [],
                "informationState": "NO_INFORMATION"
            },
            "perceivedStimuli": [],
            "nextPerceivedStimuli": []
        },
        "healthStatus": None,
        "infectionStatus": None,
        "groupIds": [],
        "groupSizes": [],
        "agentsInGroup": [],
        "trajectory": {
            "footSteps": []
        },
        "modelPedestrianMap": None,
        "type": "PEDESTRIAN"
    }

    pedestrians.append(dynamicElement)

# Assign the list of pedestrians to dynamicElements
data['scenario']['topography']['dynamicElements'] = pedestrians


# Write the updated data back to the JSON file
with open(file_path, 'w') as file:
    json.dump(data, file, indent=2)

print("File updated successfully.")
# Run the vadere-console
command = [
    "java",
    "-jar",
    "vadere-console.jar",
    "scenario-run",
    "--scenario-file",
    scenario_file,
    "--output-dir",
    output_dir
]
subprocess.run(command)
print("Output created successfully.")
