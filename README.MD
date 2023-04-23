# Fantasy Quest Prompter

This repository contains scripts and data for generating fantasy writing prompts from a given source text.

### Files:

`main.py`: A Python script that reads in a JSON file containing writing prompts, preprocesses the text, and saves the results to a new JSON file.

`requirements.txt`: A text file containing the required Python packages for running the scripts.

`data/prompts.json`: A JSON file containing a list of writing prompts.

`scripts/__init__.py`: An empty file that allows the scripts folder to be treated as a module.

`scripts/prompt_generator.py`: A Python module that defines a PromptGenerator class for generating new writing prompts.

`scripts/convert_adventure_to_prompts.py`: A Python script that reads in a text file containing a Dungeons & Dragons adventure, filters out non-prompt sentences, and generates a writing prompt for each remaining sentence.

`scripts/data_preprocessing.py`: A Python script that reads in a JSON file containing writing prompts, preprocesses the text, and saves the results to a new JSON file.

### Usage:

1. Clone the repository to your local machine.
2. Install the required Python packages by running pip install -r requirements.txt.
3. To preprocess the writing prompts, run python main.py. This will read in data/prompts.json, preprocess the text, and save the results to a new file, data/fantasy_prompts_processed.json.
4. To generate new writing prompts from a Dungeons & Dragons adventure, put the adventure text file in the same folder as convert_adventure_to_prompts.py, and run python convert_adventure_to_prompts.py. This will generate a new writing prompt for each sentence in the adventure that contains a prompt keyword and a named entity.
5. To use the PromptGenerator class to generate new writing prompts from any source text, create an instance of the class by passing the path to a JSON file containing preprocessed writing prompts as an argument. Then, call the generate_prompt method with a string containing a sentence from the source text to generate a new writing prompt.


### Roadmap:
#### Phase 1: Basic Prompt Generation

- [x] Implement a basic prompt generation algorithm
- [x] Create a command-line interface for the prompt generator
- [x] Develop a basic set of prompts for the generator to use
- [x] Add functionality for the generator to write prompts to a file
- [ ] Create documentation for the prompt generator

#### Phase 2: OpenAI Integration

- [x] Research OpenAI's GPT models and their capabilities
- [ ] Integrate OpenAI's GPT model into the prompt generation algorithm
- [ ] Create a configuration file for users to customize OpenAI's GPT model settings
- [ ] Add functionality for the generator to use OpenAI's GPT model to generate prompts
- [ ]  Test and optimize the OpenAI integration

#### Phase 3: User Interface

- [ ] Develop a web-based user interface for the prompt generator
- [ ] Add functionality for users to input custom parameters for prompt generation
- [ ] Allow users to save generated prompts to an account
- [ ] Implement user feedback and rating system for prompts

- [ ]  Add functionality for users to collaborate on prompt creation

#### Phase 4: Additional Features

- [ ] Implement sentiment analysis to ensure generated prompts are appropriate
- [ ] Add support for multiple languages
- [ ] Allow for more complex prompt generation, such as branching prompts
- [ ] Investigate integrating other AI models or techniques for prompt generation
