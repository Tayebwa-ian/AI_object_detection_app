# AI Object Counter - Image Generation Tool

## Overview

This directory contains the main image generation script for the AI Object Counter project.

## Files

- `generate_and_post.py` - Main image generation and API testing script

## Quick Start

### Generate Images and Test API

```bash
# Generate 5 images with cars and test against API
python tools/generate_and_post.py --num-images 5 --objects car --min-objects 1 --max-objects 3 --api http://localhost:5000

# Generate images with multiple object types
python tools/generate_and_post.py --num-images 10 --objects car,person,bicycle --min-objects 1 --max-objects 5

# Generate with transformations
python tools/generate_and_post.py --num-images 3 --objects car --blur --rotate --noise

# Generate with custom size
python tools/generate_and_post.py --num-images 2 --objects person --size 1024x768
```

## Command Line Arguments

- `--num-images` - Number of images to generate (required)
- `--objects` - Comma-separated object types (required, e.g., car,person,bicycle)
- `--min-objects` - Minimum objects per image (default: 1)
- `--max-objects` - Maximum objects per image (default: 5)
- `--background` - Background image path or color name (optional)
- `--blur` - Apply blur transformation
- `--rotate` - Apply rotation transformation
- `--noise` - Apply noise transformation
- `--size` - Image size as WxH (default: 800x600)
- `--api` - API base URL (default: http://localhost:5000)
- `--output-dir` - Output directory (default: tools/generated_images)

## Features

- ✅ **AI Image Generation**: Uses the university AI endpoint to generate realistic images
- ✅ **PIL Transformations**: Applies blur, rotation, noise, and size adjustments
- ✅ **API Integration**: Automatically tests images against the AI Object Counter API
- ✅ **CSV Logging**: Logs results with all required columns
- ✅ **Error Handling**: Robust error handling and logging

## Output

The script creates:
1. **Images**: `tools/generated_images/test_image_XXX.jpg`
2. **Metadata**: `tools/generated_images/test_image_XXX_metadata.json`
3. **CSV Results**: `tools/generated_images/test_results_TIMESTAMP.csv`

## CSV Output Columns

- `image_path` - Path to generated image
- `object_type` - Object type tested
- `truth` - Expected count
- `predicted` - API predicted count
- `confidence` - API confidence score
- `time_ms` - API response time
- `segments` - Number of segments found

## Requirements

- Python 3.7+
- All dependencies from `requirements.txt`
- Backend API running on specified URL
- Network access to AI endpoint (university network or VPN)
