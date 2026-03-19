# Guitar Tabs Transcriber Training

## Project Overview

**Guitar Tabs Transcriber** is a personal project I worked on in 2024. with the aim to write a deep learning model that converts guitar musics to guitar tablatures[^1].

The main goal was to build a model and a training pipeline from scratch without ML framework, **using only NumpPy**. To understand **neural networks** and **backpropagation** on a deeper level.

[^1]: Tablature (or tab for short) is a form of musical notation indicating instrument fingering or the location of the played notes rather than musical pitches.

## Data

Training data generated using [guitartabs-transcriber-data repository](https://github.com/Dnewe/guitartabs-transcriber-data).

## Training

### Model

Model architecture is defined in *src/config/modelconfig.json* at *"layers"*.

*layers* is a dictionary of *layer*, which are lists structured as follows: \
`List[<layer_type>, <in_dim>, <out_dim>, <activation>]`

- **layer_type** *(str)*: type of layer, only fully connected "fc" available.
- **in_dim** *(int)*: dimension of input vector.
- **out_dim** *(int)*: dimension of output vector.
- **activation** *(str)*: activation function "relu" or "sigmoid" or "softmax".


__Default Model__:
```json
"layers":{
    "input": ["fc", 277, 192, "relu"],
    "hidden": ["fc", 192, 128, "relu"],
    "output": ["fc", 128, 6, "sigmoid"]
}
```

> [!NOTE]
> Layer names (eg "hidden") are only used for the filenames of the corresponding layer weights.

### Hyperparameters

Hyperparameters are defined in *src/config/modelconfig.json*.\
Here is hyperparameters list:

- "**train_size**": *(default= 3e5)*: Maximum length of training dataset.
- "**val_size**": *(default= 2e4)*: Maximum length of validation dataset.
- "**epochs**": *(default= 500)*: Number of epochs.
- "**alpha**": *(default= 3e-3)*: Learning rate (i.e.: Proportion of substracting gradient (from backprop) to weight)
- "**batch_size**": *(default= 200)*: Size of training batch.
- "**earlystop_patience**" *(default= 50)*: Number of consecutive epochs without improvement of *val_loss* before stopping training.
- "**momentum**" *(default= 0.9)*: Coefficient of previous gradient added to current gradient in backpropagation algorithm.

## Install

Library requirements in *requirement.txt*.

```
python -m venv .venv
pip install -r requirements.txt
```

## Run

**Command**:
```powershell
python src/main.py --input <input_dir> --output <output_dir>
```

**Arguments:**
- **--input (-i)** *(str)*: Path to the processed input data directory. Containing *data.csv* and *metadata.json* files.
- **--output (-o)** *(str)*: Path to the directory where model weights, metamodel and run info will be written.

**Example:**
```powershell
python src/main.py -i "data/input" -o "data/output"
```

