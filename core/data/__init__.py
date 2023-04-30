from configs.dataset_cfg import VOC_CFG, COCO_CFG


def find_class_name(dataset_name: str, class_index, keep_index=False):
    if dataset_name.lower() == "voc":
        class_name_list = VOC_CFG["classes"]
    else:
        class_name_list = COCO_CFG["classes"]
    if keep_index:
        return class_name_list[class_index], class_index
    return class_name_list[class_index]