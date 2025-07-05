# Action Labeler

A Python package for labeling actions in images using AI models. This tool helps automate the process of creating labeled datasets for computer vision tasks, particularly for action recognition and object detection.

## Features

- **Dataset Management**: Load, manipulate, and save YOLO-format datasets
- **AI-Powered Labeling**: Use various AI models to automatically label actions in images
- **Multiple Model Support**: Integration with ChatGPT, Gemma, Llama, Qwen, and other vision-language models
- **Flexible Preprocessing**: Support for cropping, resizing, masking, and other image preprocessing techniques
- **Detection Filtering**: Apply various filters to refine detection results
- **Visualization**: Plot datasets and class distributions for analysis

## Installation

### Install from GitHub

```bash
pip install git+https://github.com/Jchang4/action_labeler.git
```

### Install for Development

```bash
git clone https://github.com/Jchang4/action_labeler.git
cd action_labeler
pip install -e .
```

### Install Testing Dependencies

```bash
pip install pytest pytest-cov
```

## Testing

Run tests with pytest:

```bash
# Run all tests
pytest

# Run tests with coverage report
pytest --cov=action_labeler

# Run specific test modules
pytest tests/helpers/
pytest tests/detections/

# Run tests verbosely
pytest -v
```

## Requirements

- Python >= 3.9
- ultralytics
- supervision
- pillow
- tqdm
- pyyaml
- numpy
- pandas
- matplotlib

## Quick Start

### Basic Usage

```python
import os

from pathlib import Path

from action_labeler.prompt import ActionPrompt
from action_labeler.models.llama_cpp import LlamaCpp
from action_labeler.filters import SingleDetectionFilter
from action_labeler.preprocessors import (
    BoundingBoxPreprocessor,
    ResizePreprocessor,
    CropPreprocessor,
    TextPreprocessor,
)

os.environ["OPENAI_API_KEY"] = "sk-proj-1234567890"

ACTION_PROMPT_TEMPLATE = """
Describe the actions of the dog in the image.

Output Format:
- Only respond with the action of the dog.
- Do not include any other text
- Do not provide explanations
- If none of the actions apply, respond with "none"
- If multiple actions apply, choose the most specific action.
"""


action_labeler = ActionLabeler(
    folder=Path("./samples"),
    prompt=ActionPrompt(
        template=ACTION_PROMPT_TEMPLATE,
        classes=[
            "sitting",
            "running",
            "standing",
            "walking",
            "laying down",
        ],
    ),
    model=LlamaCpp(),
    filters=[
        SingleDetectionFilter(),
    ],
    preprocessors=[
        BoundingBoxPreprocessor(),
        TextPreprocessor(),
        CropPreprocessor(),
        ResizePreprocessor(1024),
    ],
)

action_labeler.label()
```

### Model Configuration

The package supports various AI models for action labeling:

- **ChatGPT**: OpenAI's vision models
- **Gemma**: Google's Gemma models
- **Llama**: Meta's Llama models with vision capabilities
- **Qwen**: Alibaba's Qwen vision-language models
- **PaliGemma**: Google's PaliGemma model
- **Phi4**: Microsoft's Phi-4 model
- **Janus**: Janus vision-language model
- **Ovis**: Ovis vision models

## Project Structure

```
action_labeler/
├── __init__.py              # Main package exports
├── detections/              # Detection handling
├── filters/                 # Detection filters
├── helpers/                 # Utility functions
├── labeler/                 # Main labeling functionality
├── models/                  # AI model integrations
├── preprocessors/           # Image preprocessing
└── prompt/                  # Prompt templates
```

## Examples

Check out the `notebooks/` directory for detailed examples:

- `0 - Helpers.ipynb`: Utility functions and helpers
- `1 - Get Bounding Boxes.ipynb`: Object detection workflow
- `2 - Action Labeling.ipynb`: Action labeling examples

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

If you encounter any issues or have questions, please open an issue on the [GitHub repository](https://github.com/Jchang4/action_labeler/issues).
