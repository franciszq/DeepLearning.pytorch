import numpy as np


def generate_ssd_anchor(input_image_shape, anchor_sizes, feature_shapes, aspect_ratios):
    """
    生成SSD算法需要的锚框
    :param input_image_shape: 输入图片高、宽
    :param anchor_sizes:
    :param feature_shapes: 输入特征图的高、宽
    :param aspect_ratios:
    :return:
    """
    image_h, image_w = input_image_shape
    anchors = []
    for i in range(len(feature_shapes)):
        # 先验框的短边和长边
        min_size = anchor_sizes[i]
        max_size = anchor_sizes[i + 1]
        # 特征图的高和宽，它们相等
        feature_h = feature_shapes[i]
        # 对于每个像素位置，根据aspect_ratio生成不同宽、高比的先验框
        box_widths = []
        box_heights = []
        for ar in aspect_ratios[i]:
            if ar == 1:
                box_widths.append(min_size)
                box_heights.append(min_size)
                box_widths.append(np.sqrt(min_size * max_size))
                box_heights.append(np.sqrt(min_size * max_size))
            else:
                box_widths.append(min_size * np.sqrt(ar))
                box_heights.append(min_size / np.sqrt(ar))

        half_box_widths = np.array(box_widths) / 2.0  # shape: (len(aspect_ratios[i])+1,)
        half_box_heights = np.array(box_heights) / 2.0

        # 特征层上一个像素点映射到原图上对应的像素长度
        pixel_length = [image_h / feature_h, image_w / feature_h]
        # 生成网格中心
        c_x = np.linspace(0.5 * pixel_length[1], image_w - 0.5 * pixel_length[1], feature_h)
        c_y = np.linspace(0.5 * pixel_length[0], image_w - 0.5 * pixel_length[0], feature_h)
        center_x, center_y = np.meshgrid(c_x, c_y)
        center_x = np.reshape(center_x, (-1, 1))  # (feature_h**2, 1)
        center_y = np.reshape(center_y, (-1, 1))  # (feature_h**2, 1)

        anchor = np.concatenate((center_x, center_y), axis=1)  # (feature_h**2, 2)
        # 对于每一种宽高比例，都需要一个对应的先验框
        # shape: (feature_h**2, 4*(len(aspect_ratios[i])+1))
        anchor = np.tile(anchor, (1, (len(aspect_ratios[i]) + 1) * 2))

        # 转换为xmin, ymin, xmax, ymax格式
        anchor[:, ::4] -= half_box_widths  # shape: (feature_h**2, len(aspect_ratios[i])+1)
        anchor[:, 1::4] -= half_box_heights
        anchor[:, 2::4] += half_box_widths
        anchor[:, 3::4] += half_box_heights

        # 归一化
        anchor[:, ::2] /= image_w
        anchor[:, 1::2] /= image_h
        anchor = np.clip(anchor, a_min=0.0, a_max=1.0)
        anchor = np.reshape(anchor, (-1, 4))

        anchors.append(anchor)

    anchors = np.concatenate(anchors, axis=0)  # (8732, 4)
    return anchors.astype(dtype=np.float32)
