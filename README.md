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
from action_labeler import ActionLabeler

# Initialize the action labeler with a model
labeler = ActionLabeler(model_name="your_model")

# Process images and generate labels
results = labeler.label_actions("path/to/images")
```

### Advanced Usage

```python
from action_labeler import ActionLabeler

# Initialize with specific model configuration
labeler = ActionLabeler(
    model_name="gpt-4-vision",
    # Add other configuration options as needed
)

# Process a single image
result = labeler.process_image("path/to/image.jpg")

# Process multiple images
results = labeler.process_batch(["image1.jpg", "image2.jpg"])
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
