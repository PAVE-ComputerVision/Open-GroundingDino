import torch

def create_crops_v3(image_tensor, ori_tensor, bbox, padding=100, crop_size=(512, 512), stride=(256, 256)):
    """
    Create overlapping crops from the adjusted bounding box tensor.

    Parameters:
    - image_tensor: The image tensor to crop from.
    - ori_tensor: The original tensor to crop from.
    - bbox: A tensor with the format [x_min, y_min, x_max, y_max].
    - crop_size: Size of the crops (width, height).
    - stride: Step size for the sliding window (overlap control).

    Returns:
    - A list of crops, original image crops, and their respective bounding boxes.
    """
    crop_height, crop_width = crop_size
    stride_y, stride_x = stride

    x_min, y_min, x_max, y_max = bbox.tolist()  # Convert tensor to list
    pad_x_min = max(x_min - padding, 0)
    pad_y_min = max(y_min - padding, 0)
    pad_x_max = min(x_max + padding, image_tensor.shape[-1])
    pad_y_max = min(y_max + padding, image_tensor.shape[-2])
    crops = []
    ori_crops = []
    crop_bboxes = []

    pad_width = pad_x_max - pad_x_min
    num_x_crops = pad_width//stride_x + 1

    pad_height = pad_y_max - pad_y_min
    num_y_crops = pad_height//stride_y
    # Loop through the adjusted bounding box with overlap using stride
    for i in range(num_x_crops):
        for j in range(num_y_crops):
            x = pad_x_min + stride_x*i
            x_end = x + crop_width

            y = pad_y_min + stride_y*j
            y_end = y + crop_height

            if x_end > pad_x_max:
                x = pad_x_max - crop_width
                x_end = pad_x_max

            if y_end > pad_y_max:
                y = pad_y_max - crop_height
                y_end = pad_y_max

            top_left = (x, y)
            bottom_right = (x_end, y_end)

            # Crop from the image tensor
            crop = image_tensor[:, :, y:y_end, x:x_end]
            ori_crop = ori_tensor[:, y:y_end, x:x_end]

            crops.append(crop)
            ori_crops.append(ori_crop)
            crop_bboxes.append((top_left, bottom_right))

    return crops, ori_crops, crop_bboxes
