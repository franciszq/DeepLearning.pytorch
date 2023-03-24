from lib.utils.yaml_tools import load_yaml


def get_dataset_classes():
    res = dict()
    res["voc"] = load_yaml("configs/voc.yaml")
    res["coco"] = load_yaml("configs/coco.yaml")
    return res


def find_class_name(dataset_name: str, class_index, keep_index=False):
    class_name_list = get_dataset_classes()[dataset_name.lower()]["classes"]
    if keep_index:
        return class_name_list[class_index], class_index
    return class_name_list[class_index]