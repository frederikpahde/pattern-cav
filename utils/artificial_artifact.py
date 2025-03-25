from datetime import datetime
from datetime import timedelta

import numpy as np
import torch
import torchvision.transforms as T
from PIL import ImageDraw, Image


def get_artifact_kwargs(config):
    artifact_kwargs = {}
    artifact_type = config.get("artifact_type", None)
    if artifact_type == "channel":
        artifact_kwargs = {
            'op_type': config.get('op_type', 'add'),
            'channel': config.get('channel', 0),
            'value': config.get('value', 100)
        }
    elif artifact_type == "ch_time":
        artifact_kwargs = {
            "time_format": config.get("time_format", "time")
        }
    return artifact_kwargs


def insert_artifact(img, artifact_type, **kwargs):
    if artifact_type == "ch_time":
        return insert_artifact_ch_time(img, **kwargs)
    if artifact_type == "ch_text":
        return insert_artifact_ch_text(img, **kwargs)
    elif artifact_type == "channel":
        return insert_artifact_channel(img, **kwargs)
    elif artifact_type == "white_color":
        return insert_artifact_white_color(img, **kwargs)
    elif artifact_type == "red_color":
        return insert_artifact_red_color(img, **kwargs)
    else:
        raise ValueError(f"Unknown artifact_type: {artifact_type}")

def color_digit(img, **kwargs):
    assert "color_id" in kwargs
    color_id = kwargs["color_id"]
    COLOR_MAP = [
        (255, 0, 0),
        (255, 128, 0),
        (255, 255, 0),
        (0, 255, 0),
        (0, 255, 255),
        (0, 128, 255),
        (0, 0, 255),
        (127, 0, 255),
        (255, 0, 255),
        (255, 0, 127)
    ]

    color = COLOR_MAP[color_id]
    img_np = np.array(img)
    img_corrupted = (img_np * color / 255).round().astype(np.uint8)
    mask = img_np[0].round().squeeze()
    return Image.fromarray(img_corrupted), mask

def random_date(start, end):
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = np.random.randint(int_delta)
    return start + timedelta(seconds=random_second)

def insert_artifact_ch_time(img, **kwargs):
    time_format = kwargs.get("time_format", "datetime")
    time_only = time_format == "time"
    d1 = datetime.strptime('01/01/2020', '%m/%d/%Y')
    d2 = datetime.strptime('12/31/2022', '%m/%d/%Y')
    kwargs["reserved_length"] = 60 if time_only else 100
    date = random_date(d1, d2)
    if time_only:
        kwargs["min_val"] = 125
        kwargs["max_val"] = 0
        date = date.strftime("%H:%M:%S")
    kwargs["text"] = str(date)
    color = (
        np.clip(np.random.choice([10,245]) + int(np.random.normal(0, 5)), 0, 255), 
        np.clip(np.random.choice([10,245]) + int(np.random.normal(0, 5)), 0, 255), 
        np.clip(np.random.choice([10,245]) + int(np.random.normal(0, 5)), 0, 255)
    )
    kwargs["color"] = color

    return insert_artifact_ch_text(img, **kwargs)


def insert_artifact_white_color(img, **kwargs):
    img = np.array(img).astype(np.float64)
    alpha = 0.3
    img[:, :, 0] = img[:, :, 0] * (1 - alpha) + alpha * 255
    img[:, :, 1] = img[:, :, 1] * (1 - alpha) + alpha * 255
    img[:, :, 2] = img[:, :, 2] * (1 - alpha) + alpha * 255
    img = np.clip(img, 0, 255).astype(np.uint8)
    mask = torch.ones((img.shape[0], img.shape[1]))
    img = Image.fromarray(img)

    return img, mask


def insert_artifact_red_color(img, **kwargs):
    img = np.array(img).astype(np.float64)
    alpha = 0.2
    img[:, :, 0] = img[:, :, 0] * (1 - alpha) + alpha * 255
    img[:, :, 1] = img[:, :, 1] * (1 - alpha) + alpha * 0
    img[:, :, 2] = img[:, :, 2] * (1 - alpha) + alpha * 0
    img = np.clip(img, 0, 255).astype(np.uint8)
    mask = torch.ones((img.shape[0], img.shape[1]))
    img = Image.fromarray(img)
    return img, mask


def insert_artifact_channel(img, **kwargs):
    img = np.array(img).astype(np.float64)

    op_type = kwargs.get("op_type", "add")
    channel = kwargs.get("channel", 0)
    value = kwargs.get("value", 100)

    if op_type == "const":
        img[:, :, channel] = value
    elif op_type == "add":
        img[:, :, channel] += value
    elif op_type == "mul":
        img[:, :, channel] *= value
    else:
        raise ValueError(f"Unknown op_type '{op_type}', choose one of 'mul', 'add', 'const'")

    img = np.clip(img, 0, 255).astype(np.uint8)
    mask = torch.ones((img.shape[0], img.shape[1]))
    img = Image.fromarray(img)

    return img, mask


def insert_artifact_ch_text(img, **kwargs):
    text = kwargs.get("text", "Clever Hans")
    fill = kwargs.get("fill", (0, 0, 0))
    img_size = kwargs.get("img_size", 224)
    color = kwargs.get("color", (255, 255, 255))
    reserved_length = kwargs.get("reserved_length", 80)
    min_val = kwargs.get("min_val", 25)
    max_val = kwargs.get("max_val", 25)
    padding = 15

    # Random position
    end_x = img_size - reserved_length
    end_y = img_size - 20
    valid_positions = np.array([
        [padding + 5, padding + 5], 
        [padding + 5, end_y - padding - 5], 
        [end_x - padding - 5, padding + 5], 
        [end_x - padding - 5, end_y - padding - 5]
    ])
    pos = valid_positions[np.random.choice(len(valid_positions))]
    pos += np.random.normal(0, 2, 2).astype(int)
    pos[0] = np.clip(pos[0], padding, end_x - padding)
    pos[1] = np.clip(pos[1], padding, end_y - padding)

    # Random size
    size_text_img = np.random.choice(np.arange(img_size - min_val, img_size + max_val))

    # Scale pos
    scaling = size_text_img / img_size
    pos = tuple((int(pos[0] * scaling), int(pos[1] * scaling)))

    # Add Random Noise to color
    fill = tuple(np.clip(np.array(fill) + np.random.normal(0, 10, 3), 0, 255).astype(int))

    
    # Random Rotation
    rotation = np.random.choice(np.arange(-30, 31) / 10)
    image_text = Image.new('RGBA', (size_text_img, size_text_img), (0,0,0,0))
    draw = ImageDraw.Draw(image_text)
    draw.text(pos, text=text, fill=color)
    image_text = T.Resize((img_size, img_size))(image_text.rotate(rotation))

    # Insert text into image
    out = Image.composite(image_text, img, image_text)

    mask = torch.zeros((img_size, img_size))
    mask_coord = image_text.getbbox()
    mask[mask_coord[1]:mask_coord[3], mask_coord[0]:mask_coord[2]] = 1

    return out, mask
