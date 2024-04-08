# Autonomous Decision-Making

## Setup


Please create a virtual environment before running the code (see documentation for [Visual Code](https://code.visualstudio.com/docs/python/environments))

To install all dependencies run the following commands in a terminal:
```
cd code
pip install -r requirements.txt
```

## Available Maps

All available maps are provided in the folder `code/layouts` and listed in the table below.

| Map   		| File                      |
|---------------|---------------------------|
| `easy_0`      | `code/layouts/easy_0.txt` |
| `easy_1`      | `code/layouts/easy_1.txt` |
| `medium_0`    | `code/layouts/medium_0.txt` |
| `medium_1`    | `code/layouts/medium_1.txt` |
| `hard_0`      | `code/layouts/hard_0.txt` |
| `hard_1`      | `code/layouts/hard_1.txt` |


## Usage

Run agent using the following commands in a terminal (`map-name` is provided in the "Map"-column of the table above):
```
cd code
python main.py <map-name>
```

## results

Easy_1 map:

https://github.com/unforgettablexD/Autonomous-decision-making-maze-game-/assets/49744841/3a75d903-6edc-4a4a-85b8-f7e28f272b7f

Medium_0 map:

https://github.com/unforgettablexD/Autonomous-decision-making-maze-game-/assets/49744841/44875ec3-4211-4e88-8002-50528456bc3c

hard_0 map:

https://github.com/unforgettablexD/Autonomous-decision-making-maze-game-/assets/49744841/2f5e5426-3802-44a6-b152-345a1ff19876


Medium_1 map:

https://github.com/unforgettablexD/Autonomous-decision-making-maze-game-/assets/49744841/30d67dab-c630-4f58-917f-a0935be9ccb5


## Compare

```
cd code
python compare.py <map-name>
```
### Simple TDLambda
![smoothed_agent_performance_comparison](https://github.com/unforgettablexD/Autonomous-decision-making-maze-game-/assets/49744841/1a9e841a-6797-4dc6-9746-30398aaf117d)


### Modified TDLambda
![combined_agent_performance](https://github.com/unforgettablexD/Autonomous-decision-making-maze-game-/assets/49744841/2351c06b-1a05-4dc4-a8cd-996cd279509d)


