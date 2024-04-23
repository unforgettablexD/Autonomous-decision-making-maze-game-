To convert the provided text into a well-formatted Markdown document suitable for a GitHub README or similar documentation, you can use the following template. This includes code blocks for commands and code snippets, along with appropriate headers and sections.

```markdown
# CSCI 599 Final Course Project

This repository contains the implementation of an autonomous agent that solves a grid-based navigation problem.

## Usage

### Creating a virtual environment and installing requirements

We recommend creating a virtual environment and installing the requirements as specified in the `requirements.txt` file.

```bash
python -m venv path/to/env
```

To activate the virtual environment, perform the following:

```bash
source path/to/env/bin/activate
```

Navigate to the code directory and enter the following:

```bash
pip install -r requirements.txt
```

### Training and Testing the Model

Our model is stored in a file called `agent.pkl` and it can be accessed using the following snippet. This agent is trained on the `hard_0` map. Testing is also done on the same.

```python
with open('agent.pkl', 'rb') as file:
    agent = pickle.load(file)
```

Ensure that you put the agent into test mode as follows:

```python
agent.test_mode()
```

To train our model, run the main file as follows:

```bash
python main.py hard_0
```

Replace `hard_0` with a map of your choice.

To test our model, run the following test.py as follows:

```bash
python test.py hard_0
```

You may replace `hard_0` with any other map, however, ensure that the pickle file used opens and agent trained on the same map.

The `test.py` file tests our agent on every occupiable position of the map. It also tests the agent on 10 random positions as specified as the testing requirements in class.
```
