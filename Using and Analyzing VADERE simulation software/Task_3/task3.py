import json
import numpy as np
import subprocess
# Specify the path to your JSON file
file_path = 'scenarios/Corner.scenario'

# Read the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)

true = True
false = False
null = None

x_position = 13.6
y_position = 2.1

scenario_file = "scenarios/Corner.scenario"
output_dir = "console_output"



target_idss = []

for i in range(0, len(data['scenario']['topography']['targets'])):
    target_id = data['scenario']['topography']['targets'][i]['id']
    target_idss.append(target_id)

dynamicElement = [{
    "attributes" : {
      "id" : 22,
      "shape" : {
        "x" : 0.0,
        "y" : 0.0,
        "width" : 1.0,
        "height" : 1.0,
        "type" : "RECTANGLE"
      },
      "visible" : true,
      "radius" : 0.2,
      "densityDependentSpeed" : false,
      "speedDistributionMean" : 1.34,
      "speedDistributionStandardDeviation" : 0.26,
      "minimumSpeed" : 0.5,
      "maximumSpeed" : 2.2,
      "acceleration" : 2.0,
      "footstepHistorySize" : 4,
      "searchRadius" : 1.0,
      "walkingDirectionSameIfAngleLessOrEqual" : 45.0,
      "walkingDirectionCalculation" : "BY_TARGET_CENTER"
    },
    "source" : null,
    "targetIds" : target_idss,
    "nextTargetListIndex" : 0,
    "isCurrentTargetAnAgent" : false,
    "position" : {
      "x" : x_position,
      "y" : y_position
    },
    "velocity" : {
      "x" : 0.0,
      "y" : 0.0
    },
    "freeFlowSpeed" : 1.3653335527746429,
    "followers" : [ ],
    "idAsTarget" : -1,
    "isChild" : false,
    "isLikelyInjured" : false,
    "psychologyStatus" : {
      "mostImportantStimulus" : null,
      "threatMemory" : {
        "allThreats" : [ ],
        "latestThreatUnhandled" : false
      },
      "selfCategory" : "TARGET_ORIENTED",
      "groupMembership" : "OUT_GROUP",
      "knowledgeBase" : {
        "knowledge" : [ ],
        "informationState" : "NO_INFORMATION"
      },
      "perceivedStimuli" : [ ],
      "nextPerceivedStimuli" : [ ]
    },
    "healthStatus" : null,
    "infectionStatus" : null,
    "groupIds" : [ ],
    "groupSizes" : [ ],
    "agentsInGroup" : [ ],
    "trajectory" : {
      "footSteps" : [ ]
    },
    "modelPedestrianMap" : null,
    "type" : "PEDESTRIAN"
  }]

data['scenario']['topography']['dynamicElements'] = dynamicElement

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
