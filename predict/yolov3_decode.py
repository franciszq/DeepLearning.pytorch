import torch


def predict_bounding_bbox(num_classes, feature_map, anchors, device, is_training=False):
    N, C, H, W = feature_map.size()
    feature_map = torch.permute(feature_map, dims=(0, 2, 3, 1))
    anchors = torch.reshape(anchors, shape=(1, 1, 1, -1, 2))
    grid_y = torch.reshape(torch.arange(0, H, dtype=torch.float32, device=device), (-1, 1, 1, 1))
    grid_y = torch.tile(grid_y, dims=(1, W, 1, 1))
    grid_x = torch.reshape(torch.arange(0, W, dtype=torch.float32, device=device), (1, -1, 1, 1))
    grid_x = torch.tile(grid_x, dims=(H, 1, 1, 1))
    grid = torch.cat((grid_x, grid_y), dim=-1)
    feature_map = torch.reshape(feature_map, shape=(-1, H, W, 3, num_classes + 5))
    box_xy = (torch.sigmoid(feature_map[..., 0:2]) + grid) / H
    box_wh = torch.exp(feature_map[..., 2:4]) * anchors
    confidence = torch.sigmoid(feature_map[..., 4:5])
    class_prob = torch.sigmoid(feature_map[..., 5:])
    if is_training:
        return box_xy, box_wh, grid, feature_map
    else:
        return box_xy, box_wh, confidence, class_prob