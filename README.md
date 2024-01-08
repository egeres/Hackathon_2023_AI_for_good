# AI Biases Analyzer
Hackathon for Good 2023

## Installation

TODO: Add installation instructions

## Running the Analyzer

### Command line interface
the file *main.py* contains the command line interface for the analyzer built with [Typer](https://github.com/tiangolo/typer). To run it, use the following command.

#### Run

```bash
python main.py run
```

Runs the entire end-to-end pipeline, generating images and evaluating them. It can take some arguments:

```
┌─ Options ───────────────────────────────────────────────────────────────────┐
│ --prompts           TEXT     [default: None]                                │
│ --n-images          INTEGER  [default: None]                                │
│ --iterations        INTEGER  [default: None]                                │
│ --help                       Show this message and exit.                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Generate

```bash
python main.py generate
```

Generates images from a given set of prompts.

```
┌─ Options ───────────────────────────────────────────────────────────────────┐
│ --prompts           TEXT     [default: None]                                │
│ --n-images          INTEGER  [default: None]                                │
│ --iterations        INTEGER  [default: None]                                │
│ --help                       Show this message and exit.                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Evaluate

```bash
python main.py evaluate
```

Executes the evaluation pipeline over the generated images on specified prompts.

```
┌─ Arguments ─────────────────────────────────────────────────────────────────┐
│ *    prompts      PROMPTS...  [default: None] [required]                    │
└─────────────────────────────────────────────────────────────────────────────┘
┌─ Options ───────────────────────────────────────────────────────────────────┐
│ --help          Show this message and exit.                                 │
└─────────────────────────────────────────────────────────────────────────────┘

```


### Streamlit App

To start streamlit app run

```bash
streamlit run app/app.py
```